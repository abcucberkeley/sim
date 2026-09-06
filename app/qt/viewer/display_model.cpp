#include "qt/viewer/display_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/array_source.hpp"

namespace sirius::app {

    namespace {
        // Robust window from the [0.1, 99.9] percentiles of a sub-sample.
        DisplayWindow robustWindow(const std::vector<float>& samples) {
            std::vector<float> s;
            s.reserve(samples.size());
            for (float v : samples)
                if (!std::isnan(v)) s.push_back(v);
            if (s.empty()) return {0.0f, 1.0f};
            auto at = [&](double frac) {
                const std::ptrdiff_t k =
                    static_cast<std::ptrdiff_t>(std::llround(frac * static_cast<double>(s.size() - 1)));
                std::nth_element(s.begin(), s.begin() + k, s.end());
                return s[static_cast<std::size_t>(k)];
            };
            float lo = at(0.001), hi = at(0.999);
            if (!(hi > lo)) {
                const auto mm = std::minmax_element(s.begin(), s.end());
                lo = *mm.first;
                hi = *mm.second;
            }
            if (!(hi > lo)) hi = lo + 1.0f;
            return {lo, hi};
        }

        inline std::uint32_t packRgb(int r, int g, int b) {
            return 0xff000000u | (static_cast<std::uint32_t>(std::min(r, 255)) << 16) |
                   (static_cast<std::uint32_t>(std::min(g, 255)) << 8) | static_cast<std::uint32_t>(std::min(b, 255));
        }
    } // namespace

    void DisplayModel::setOutput(std::shared_ptr<const StepOutput> out) {
        if (out == out_) return;
        const bool sameShape = out && out_ && out->meta.dims == out_->meta.dims && out->meta.rgb == out_->meta.rgb;
        out_ = std::move(out);
        meta_ = out_ ? out_->meta : DatasetMeta{};
        volumes_.clear();
        mips_.clear();
        planes_.clear();
        tooLarge_ = false;
        // A re-run of the same step keeps the user's windows; a different
        // shape (another step) starts over.
        if (!sameShape) windows_.clear();
    }

    bool DisplayModel::valid() const noexcept {
        return out_ && (out_->array || out_->source) && meta_.dims.numel() > 0;
    }

    bool DisplayModel::hasLabels() const noexcept { return out_ && out_->labels && !out_->labels->empty(); }

    const LabelVolume* DisplayModel::labels() const noexcept { return hasLabels() ? out_->labels.get() : nullptr; }

    // --- windows -------------------------------------------------------------

    DisplayWindow DisplayModel::window(Index c, Index t) {
        auto it = windows_.find(c);
        if (it != windows_.end()) return it->second;
        const DisplayWindow w = computeWindow(c, t);
        windows_[c] = w;
        return w;
    }

    void DisplayModel::setWindow(Index c, DisplayWindow w) { windows_[c] = w; }

    void DisplayModel::resetWindows() { windows_.clear(); }

    DisplayWindow DisplayModel::computeWindow(Index c, Index t) {
        if (!valid()) return {0.0f, 1.0f};
        const Dims5& d = meta_.dims;
        std::vector<float> samples;
        constexpr Index kTarget = 1 << 18;
        // a handful of planes spread over z, each sub-sampled at a fixed stride
        const Index nPlanes = std::min<Index>(d.z, 5);
        const Index perPlane = std::max<Index>(1, kTarget / nPlanes);
        const Index stride = std::max<Index>(1, d.planeSize() / perPlane);
        for (Index k = 0; k < nPlanes; ++k) {
            const Index z = nPlanes == 1 ? 0 : k * (d.z - 1) / (nPlanes - 1);
            const float* p = plane(c, t, z);
            if (!p) continue;
            for (Index i = 0; i < d.planeSize(); i += stride) samples.push_back(p[i]);
        }
        return robustWindow(samples);
    }

    // --- data ------------------------------------------------------------------

    const float* DisplayModel::plane(Index c, Index t, Index z) {
        if (!valid()) return nullptr;
        const Dims5& d = meta_.dims;
        if (c < 0 || c >= d.c || t < 0 || t >= d.t || z < 0 || z >= d.z) return nullptr;
        if (out_->array) return out_->array->plane(c, t, z);
        // a cached volume serves planes without touching the disk
        auto vit = volumes_.find(Key{c, t});
        if (vit != volumes_.end()) return vit->second.data() + z * d.planeSize();
        const PlaneKey key{c, t, z};
        auto it = planes_.find(c);
        if (it != planes_.end() && it->second.first == key) return it->second.second.data();
        Buffer<float> buf(Shape{d.y, d.x});
        try {
            out_->source->readPlane(c, t, z, buf.data());
        } catch (const std::exception&) {
            return nullptr;
        }
        auto& slot = planes_[c];
        slot.first = key;
        slot.second = std::move(buf);
        return slot.second.data();
    }

    const float* DisplayModel::volume(Index c, Index t) {
        if (!valid()) return nullptr;
        const Dims5& d = meta_.dims;
        if (c < 0 || c >= d.c || t < 0 || t >= d.t) return nullptr;
        if (out_->array) return out_->array->plane(c, t, 0);
        const Key key{c, t};
        auto it = volumes_.find(key);
        if (it != volumes_.end()) return it->second.data();
        const std::size_t bytes = static_cast<std::size_t>(d.z * d.planeSize()) * sizeof(float);
        if (bytes > kVolumeCacheLimit) {
            tooLarge_ = true;
            return nullptr;
        }
        // keep at most the volumes of one time point per channel
        for (auto vit = volumes_.begin(); vit != volumes_.end();)
            vit = vit->first.t != t ? volumes_.erase(vit) : std::next(vit);
        for (auto mit = mips_.begin(); mit != mips_.end();)
            mit = mit->first.t != t ? mips_.erase(mit) : std::next(mit);
        Buffer<float> buf(Shape{d.z, d.y, d.x});
        try {
            out_->source->readVolume(c, t, buf.data());
        } catch (const std::exception&) {
            return nullptr;
        }
        tooLarge_ = false;
        return volumes_.emplace(key, std::move(buf)).first->second.data();
    }

    const float* DisplayModel::mip(Index c, Index t) {
        const Key key{c, t};
        auto it = mips_.find(key);
        if (it != mips_.end()) return it->second.data();
        const float* v = volume(c, t);
        if (!v) return nullptr;
        const Dims5& d = meta_.dims;
        Buffer<float> m(Shape{d.y, d.x});
        const Index n = d.planeSize();
        std::copy_n(v, n, m.data());
        for (Index z = 1; z < d.z; ++z) {
            const float* p = v + z * n;
            float* o = m.data();
            for (Index i = 0; i < n; ++i)
                if (p[i] > o[i]) o[i] = p[i];
        }
        return mips_.emplace(key, std::move(m)).first->second.data();
    }

    bool DisplayModel::volumeTooLarge() const noexcept { return tooLarge_; }

    std::optional<float> DisplayModel::valueAt(Index c, Index t, Index z, Index y, Index x) {
        const Dims5& d = meta_.dims;
        if (!valid() || y < 0 || y >= d.y || x < 0 || x >= d.x) return std::nullopt;
        const float* p = plane(c, t, z);
        if (!p) return std::nullopt;
        return p[y * d.x + x];
    }

    void DisplayModel::dropVolumeCaches() {
        volumes_.clear();
        mips_.clear();
        planes_.clear();
    }

    // --- rendering ---------------------------------------------------------------

    std::array<int, 3> DisplayModel::tintOf(const DatasetMeta& m, Index c, bool rgb) {
        if (rgb) {
            std::array<int, 3> t{0, 0, 0};
            if (c >= 0 && c < 3) t[static_cast<std::size_t>(c)] = 256;
            return t;
        }
        if (c < 0 || static_cast<std::size_t>(c) >= m.channels.size()) return {256, 256, 256};
        const auto& col = m.channels[static_cast<std::size_t>(c)].color;
        return {static_cast<int>(std::lround(std::clamp(col[0], 0.f, 1.f) * 256)),
                static_cast<int>(std::lround(std::clamp(col[1], 0.f, 1.f) * 256)),
                static_cast<int>(std::lround(std::clamp(col[2], 0.f, 1.f) * 256))};
    }

    std::vector<DisplayModel::ChannelPlane> DisplayModel::visibleChannels(const ViewState& vs, Index t) {
        std::vector<ChannelPlane> chans;
        const Dims5& d = meta_.dims;
        for (Index c = 0; c < d.c; ++c) {
            if (!vs.channelOn(c)) continue;
            ChannelPlane cp;
            cp.tint = tintOf(meta_, c, meta_.rgb);
            cp.window = window(c, t);
            chans.push_back(cp);
        }
        return chans;
    }

    void DisplayModel::blend(std::vector<ChannelPlane> chans, Index rows, Index cols, int factor, QImage& img) {
        factor = std::max(factor, 1);
        const int w = static_cast<int>((cols + factor - 1) / factor);
        const int h = static_cast<int>((rows + factor - 1) / factor);
        if (img.width() != w || img.height() != h || img.format() != QImage::Format_RGB32)
            img = QImage(std::max(w, 1), std::max(h, 1), QImage::Format_RGB32);
        bool any = false;
        for (const ChannelPlane& cp : chans) any = any || cp.data;
        if (!any) {
            img.fill(QColor(0x0a, 0x09, 0x09));
            return;
        }
        // 2 x 2 mini-average when sub-sampling keeps noisy data from sparkling
        const int sub = factor >= 2 ? 2 : 1;
        for (int y = 0; y < h; ++y) {
            std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(img.scanLine(y));
            const Index y0 = static_cast<Index>(y) * factor;
            for (int x = 0; x < w; ++x) {
                const Index x0 = static_cast<Index>(x) * factor;
                int r = 0, g = 0, b = 0;
                for (const ChannelPlane& cp : chans) {
                    if (!cp.data) continue;
                    float v;
                    if (sub == 1) {
                        v = cp.data[y0 * cp.rowStride + x0 * cp.colStride];
                    } else {
                        const Index y1 = std::min(y0 + 1, rows - 1), x1 = std::min(x0 + 1, cols - 1);
                        v = 0.25f * (cp.data[y0 * cp.rowStride + x0 * cp.colStride] +
                                     cp.data[y0 * cp.rowStride + x1 * cp.colStride] +
                                     cp.data[y1 * cp.rowStride + x0 * cp.colStride] +
                                     cp.data[y1 * cp.rowStride + x1 * cp.colStride]);
                    }
                    const float scale = 255.0f / (cp.window.hi - cp.window.lo);
                    const float gf = (v - cp.window.lo) * scale;
                    // NaN compares false on both sides and maps to 0
                    const int gi = gf > 255.0f ? 255 : (gf > 0.0f ? static_cast<int>(gf) : 0);
                    r += (gi * cp.tint[0]) >> 8;
                    g += (gi * cp.tint[1]) >> 8;
                    b += (gi * cp.tint[2]) >> 8;
                }
                dst[x] = packRgb(r, g, b);
            }
        }
    }

    void DisplayModel::renderXY(Index t, Index z, const ViewState& vs, int factor, QImage& img) {
        const Dims5& d = meta_.dims;
        std::vector<ChannelPlane> chans = visibleChannels(vs, t);
        Index k = 0;
        for (Index c = 0; c < d.c; ++c) {
            if (!vs.channelOn(c)) continue;
            chans[static_cast<std::size_t>(k)].data = plane(c, t, z);
            chans[static_cast<std::size_t>(k)].rowStride = d.x;
            ++k;
        }
        blend(std::move(chans), d.y, d.x, factor, img);
    }

    void DisplayModel::renderXZ(Index t, Index y, const ViewState& vs, QImage& img) {
        const Dims5& d = meta_.dims;
        std::vector<ChannelPlane> chans = visibleChannels(vs, t);
        Index k = 0;
        y = std::clamp<Index>(y, 0, d.y - 1);
        for (Index c = 0; c < d.c; ++c) {
            if (!vs.channelOn(c)) continue;
            const float* v = volume(c, t);
            chans[static_cast<std::size_t>(k)].data = v ? v + y * d.x : nullptr;
            chans[static_cast<std::size_t>(k)].rowStride = d.y * d.x;   // next z
            ++k;
        }
        blend(std::move(chans), d.z, d.x, 1, img);
    }

    void DisplayModel::renderYZ(Index t, Index x, const ViewState& vs, QImage& img) {
        const Dims5& d = meta_.dims;
        std::vector<ChannelPlane> chans = visibleChannels(vs, t);
        x = std::clamp<Index>(x, 0, d.x - 1);
        // gather the column of every visible channel into (c, y, z) scratch
        sliceScratch_.resize(chans.size() * static_cast<std::size_t>(d.y * d.z) + 1);
        Index k = 0;
        for (Index c = 0; c < d.c; ++c) {
            if (!vs.channelOn(c)) continue;
            const float* v = volume(c, t);
            float* dst = sliceScratch_.data() + k * d.y * d.z;
            if (v) {
                for (Index z = 0; z < d.z; ++z) {
                    const float* src = v + z * d.y * d.x + x;
                    for (Index yy = 0; yy < d.y; ++yy) dst[yy * d.z + z] = src[yy * d.x];
                }
                chans[static_cast<std::size_t>(k)].data = dst;
            } else {
                chans[static_cast<std::size_t>(k)].data = nullptr;
            }
            chans[static_cast<std::size_t>(k)].rowStride = d.z;
            ++k;
        }
        blend(std::move(chans), d.y, d.z, 1, img);
    }

    void DisplayModel::renderMIP(Index t, const ViewState& vs, int factor, QImage& img) {
        const Dims5& d = meta_.dims;
        std::vector<ChannelPlane> chans = visibleChannels(vs, t);
        Index k = 0;
        for (Index c = 0; c < d.c; ++c) {
            if (!vs.channelOn(c)) continue;
            chans[static_cast<std::size_t>(k)].data = mip(c, t);
            chans[static_cast<std::size_t>(k)].rowStride = d.x;
            ++k;
        }
        blend(std::move(chans), d.y, d.x, factor, img);
    }

    // --- labels ----------------------------------------------------------------------

    void DisplayModel::overlay(const std::uint32_t* lab, Index rows, Index cols, Index rowStride, Index colStride,
                               int factor, float opacity, std::uint32_t selected, QImage& img) {
        if (!lab || img.isNull()) return;
        factor = std::max(factor, 1);
        const int w = img.width(), h = img.height();
        // sub-sampled label plane, so outlines are computed at display resolution
        labelScratch_.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        for (int y = 0; y < h; ++y) {
            const Index sy = std::min<Index>(static_cast<Index>(y) * factor, rows - 1);
            for (int x = 0; x < w; ++x) {
                const Index sx = std::min<Index>(static_cast<Index>(x) * factor, cols - 1);
                labelScratch_[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)] =
                    lab[sy * rowStride + sx * colStride];
            }
        }
        const int fill = static_cast<int>(std::lround(std::clamp(opacity, 0.0f, 1.0f) * 256));
        const int edge = 230;
        auto at = [&](int y, int x) { return labelScratch_[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; };
        for (int y = 0; y < h; ++y) {
            std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(img.scanLine(y));
            for (int x = 0; x < w; ++x) {
                const std::uint32_t id = at(y, x);
                if (id == 0) continue;
                const bool border = (x > 0 && at(y, x - 1) != id) || (x + 1 < w && at(y, x + 1) != id) ||
                                    (y > 0 && at(y - 1, x) != id) || (y + 1 < h && at(y + 1, x) != id);
                const std::array<float, 3> col = labelColor(id);
                int cr = static_cast<int>(col[0] * 255), cg = static_cast<int>(col[1] * 255),
                    cb = static_cast<int>(col[2] * 255);
                int a = border ? edge : fill;
                if (selected != 0 && id == selected) {
                    a = border ? 256 : std::min(256, fill + 80);
                    if (border) cr = cg = cb = 255;   // selected label: white outline
                }
                const std::uint32_t p = dst[x];
                const int r = static_cast<int>((p >> 16) & 0xff), g = static_cast<int>((p >> 8) & 0xff),
                          b = static_cast<int>(p & 0xff);
                dst[x] = packRgb((r * (256 - a) + cr * a) >> 8, (g * (256 - a) + cg * a) >> 8,
                                 (b * (256 - a) + cb * a) >> 8);
            }
        }
    }

    void DisplayModel::overlayLabelsXY(Index t, Index z, int factor, const ViewState& vs, QImage& img) {
        const LabelVolume* L = labels();
        if (!L || t >= L->t() || z >= L->z()) return;
        overlay(L->plane(t, z), L->y(), L->x(), L->x(), 1, factor, static_cast<float>(vs.labelOpacity),
                vs.selectedLabel, img);
    }

    void DisplayModel::overlayLabelsXZ(Index t, Index y, const ViewState& vs, QImage& img) {
        const LabelVolume* L = labels();
        if (!L || t >= L->t() || y >= L->y()) return;
        overlay(L->volume(t) + y * L->x(), L->z(), L->x(), L->y() * L->x(), 1, 1,
                static_cast<float>(vs.labelOpacity), vs.selectedLabel, img);
    }

    void DisplayModel::overlayLabelsYZ(Index t, Index x, const ViewState& vs, QImage& img) {
        const LabelVolume* L = labels();
        if (!L || t >= L->t() || x >= L->x()) return;
        // rows y, cols z: element (y, z) = volume[(z * Y + y) * X + x]
        overlay(L->volume(t) + x, L->y(), L->z(), L->x(), L->y() * L->x(), 1,
                static_cast<float>(vs.labelOpacity), vs.selectedLabel, img);
    }

} // namespace sirius::app
