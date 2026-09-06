// The worker protocol: frame encoding, a scripted worker on the loopback
// transport answering hello / run / cancel, progress streaming and error
// propagation. The Python worker implements the same bytes; its own tests
// live in app/python/tests.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <thread>

#include "core/rpc.hpp"

using namespace sirius;
using namespace sirius::app;
using json = nlohmann::json;

namespace {

    // Minimal worker: answers on its own thread until the transport closes.
    struct ScriptedWorker {
        std::unique_ptr<rpc::Transport> t;
        std::thread thread;
        std::atomic<bool> stop{false};
        std::string expectedToken;
        bool slow = false;

        explicit ScriptedWorker(std::unique_ptr<rpc::Transport> transport, std::string token = {}, bool slowRun = false)
            : t(std::move(transport)), expectedToken(std::move(token)), slow(slowRun) {
            thread = std::thread([this] { loop(); });
        }
        ~ScriptedWorker() {
            stop = true;
            t->close();
            thread.join();
        }
        void send(const json& h, const std::vector<rpc::TensorRef>& tensors = {}) { t->send(rpc::encodeFrame(h, tensors)); }
        void loop() {
            std::vector<std::byte> buf;
            std::atomic<bool> cancelled{false};
            try {
                while (!stop) {
                    auto m = rpc::decodeFrame(buf);
                    if (!m) {
                        t->receive(buf, std::chrono::milliseconds(50));
                        continue;
                    }
                    const json& h = m->header;
                    const std::uint64_t id = h.value("id", 0ull);
                    const std::string method = h.value("method", "");
                    if (method == "hello") {
                        if (!expectedToken.empty() && h.value("token", "") != expectedToken) {
                            send({{"id", id}, {"type", "error"}, {"message", "bad token"}});
                            continue;
                        }
                        send({{"id", id}, {"type", "result"}, {"result", {{"version", "test"}, {"methods", {"run:torch_segment", "model_info"}}, {"cuda", false}, {"device", "cpu"}, {"hostname", "loop"}}}});
                    } else if (method == "cancel") {
                        cancelled = true;
                    } else if (method == "run") {
                        const std::string kind = h["params"].value("kind", "");
                        if (kind == "fail") {
                            send({{"id", id}, {"type", "error"}, {"message", "kaboom"}});
                            continue;
                        }
                        REQUIRE(m->tensors.size() == 1);
                        const rpc::Tensor& in = m->tensors[0];
                        cancelled = false;
                        for (int i = 0; i < (slow ? 40 : 3); ++i) {
                            // keep servicing cancel requests while "working"
                            auto c = rpc::decodeFrame(buf);
                            if (!c) t->receive(buf, std::chrono::milliseconds(slow ? 25 : 1));
                            else if (c->header.value("method", "") == "cancel") cancelled = true;
                            if (cancelled) break;
                            send({{"id", id}, {"type", "progress"}, {"fraction", (i + 1) / 3.0}, {"message", "tile " + std::to_string(i)}});
                        }
                        if (cancelled) {
                            send({{"id", id}, {"type", "error"}, {"message", "cancelled"}});
                            continue;
                        }
                        std::vector<float> out(static_cast<std::size_t>(in.numel()));
                        const float* src = in.asFloat32();
                        for (std::size_t i = 0; i < out.size(); ++i) out[i] = src[i] * 2.0f;
                        rpc::TensorRef ref{"prob", "float32", in.shape, out.data(), out.size() * sizeof(float)};
                        send({{"id", id}, {"type", "result"}, {"result", {{"channels", 1}}}}, {ref});
                    } else {
                        send({{"id", id}, {"type", "error"}, {"message", "unknown method " + method}});
                    }
                }
            } catch (const std::exception&) {
                // the client closed: done
            }
        }
    };

} // namespace

TEST_CASE("rpc frames round trip with tensors", "[app][rpc]") {
    std::vector<float> a{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<std::uint32_t> b{7u, 8u};
    std::vector<rpc::TensorRef> refs{{"a", "float32", {2, 3}, a.data(), a.size() * sizeof(float)},
                                     {"b", "uint32", {2}, b.data(), b.size() * sizeof(std::uint32_t)}};
    const json header = {{"id", 5}, {"type", "request"}, {"method", "run"}, {"params", {{"kind", "x"}}}};
    std::vector<std::byte> bytes = rpc::encodeFrame(header, refs);
    CHECK(bytes.size() > 4 + 8 + 24 + 8);

    SECTION("a partial buffer yields nothing and keeps its bytes") {
        std::vector<std::byte> part(bytes.begin(), bytes.begin() + 10);
        CHECK_FALSE(rpc::decodeFrame(part));
        CHECK(part.size() == 10);
    }
    SECTION("two frames back to back decode one at a time") {
        std::vector<std::byte> two = bytes;
        two.insert(two.end(), bytes.begin(), bytes.end());
        auto m1 = rpc::decodeFrame(two);
        REQUIRE(m1);
        CHECK(m1->header["method"] == "run");
        REQUIRE(m1->tensors.size() == 2);
        CHECK(m1->tensors[0].shape == std::vector<Index>{2, 3});
        CHECK(m1->tensors[0].asFloat32()[5] == 6.f);
        CHECK(m1->tensors[1].asUInt32()[1] == 8u);
        CHECK_THROWS(m1->tensors[1].asFloat32());
        auto m2 = rpc::decodeFrame(two);
        REQUIRE(m2);
        CHECK(two.empty());
    }
    SECTION("size mismatches are rejected") {
        std::vector<rpc::TensorRef> bad{{"a", "float32", {2, 2}, a.data(), a.size() * sizeof(float)}};
        CHECK_THROWS(rpc::encodeFrame(header, bad));
        std::vector<std::byte> garbage(20, std::byte{0xff});
        CHECK_THROWS(rpc::decodeFrame(garbage));
    }
}

TEST_CASE("RemoteWorker talks to a scripted worker over the loopback", "[app][rpc]") {
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server), "secret");

    SECTION("a wrong token is refused") {
        CHECK_THROWS_WITH(RemoteWorker(std::move(client), "nope"), Catch::Matchers::ContainsSubstring("bad token"));
    }
    SECTION("hello, run with progress, error") {
        RemoteWorker rw(std::move(client), "secret");
        CHECK(rw.capabilities().version == "test");
        CHECK(rw.supports("torch_segment"));
        CHECK_FALSE(rw.supports("sim"));
        std::vector<float> in{1.f, 2.f, 3.f, 4.f};
        std::vector<double> fractions;
        WorkerResult r = rw.call("run", {{"kind", "torch_segment"}}, {{"input", "float32", {2, 2}, in.data(), 16}},
                                 [&](double f, const std::string&) { fractions.push_back(f); });
        CHECK(fractions.size() == 3);
        REQUIRE(r.tensors.size() == 1);
        CHECK(r.tensors[0].name == "prob");
        CHECK(r.tensors[0].asFloat32()[3] == 8.f);
        CHECK(r.result["channels"] == 1);
        CHECK_THROWS_WITH(rw.call("run", {{"kind", "fail"}}, {{"input", "float32", {1}, in.data(), 4}}),
                          Catch::Matchers::ContainsSubstring("kaboom"));
        rw.close();
        CHECK_FALSE(rw.isOpen());
    }
}

TEST_CASE("RemoteWorker cancels a slow request", "[app][rpc]") {
    auto [client, server] = rpc::loopbackPair();
    ScriptedWorker worker(std::move(server), "", true);
    RemoteWorker rw(std::move(client));
    std::vector<float> in{1.f};
    std::atomic<bool> cancel{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        cancel = true;
    });
    CHECK_THROWS_WITH(rw.call("run", {{"kind", "torch_segment"}}, {{"input", "float32", {1}, in.data(), 4}}, {},
                              [&] { return cancel.load(); }),
                      Catch::Matchers::ContainsSubstring("cancelled"));
    canceller.join();
}

TEST_CASE("connectTcp reports an unreachable port", "[app][rpc]") {
    CHECK_THROWS(rpc::connectTcp("127.0.0.1", 1, std::chrono::milliseconds(500)));
    (void)workerScriptPath("/definitely/not/here");   // must not throw
}
