// Public Eigen-tensor front end of the preprocessing stages. The loops
// themselves live in sim_cpu_stages.hpp and are shared with the CPU backend of
// the reconstruction pipeline, so this file only adapts the (ndirs, nphases,
// nz, ny, nx) tensor layout to the flat (nsec, ny, nx) section view.

#include "sirius/preprocess.hpp"

#include "sim_cpu_stages.hpp"

#include <vector>

namespace sirius {

    namespace {
        using Stack5 = Eigen::Tensor<double, 5, Eigen::RowMajor>;
        using simdetail::IndexT;

        struct SectionLayout {
            IndexT ndirs, nphases, nz, ny, nx;
            IndexT nsec() const { return ndirs * nphases * nz; }
            IndexT sec() const { return ny * nx; }
        };

        SectionLayout layoutOf(const Stack5& data) {
            return {data.dimension(0), data.dimension(1), data.dimension(2),
                    data.dimension(3), data.dimension(4)};
        }
    } // namespace

    void bleach_rescale(Stack5& data, bool equalizez) {
        const SectionLayout l = layoutOf(data);
        if (data.size() == 0) return;

        std::vector<double> sums(static_cast<std::size_t>(l.nsec()));
        std::vector<double> factors(sums.size());
        simdetail::cpu::planeSums(data.data(), l.nsec(), l.sec(), sums.data());
        simdetail::cpu::bleachFactors(sums.data(), l.ndirs, l.nphases, l.nz, equalizez, factors.data());
        simdetail::cpu::scalePlanes(data.data(), l.nsec(), l.sec(), factors.data());
    }

    void edge_apodization(Stack5& data, int napodize) {
        const SectionLayout l = layoutOf(data);
        if (data.size() == 0) return;
        simdetail::cpu::edgeApodize(data.data(), l.nsec(), l.ny, l.nx, napodize);
    }

    void cosine_apodization(Stack5& data) {
        const SectionLayout l = layoutOf(data);
        if (data.size() == 0) return;
        simdetail::cpu::cosineApodize(data.data(), l.nsec(), l.ny, l.nx);
    }

} // namespace sirius
