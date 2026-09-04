#ifndef SIRIUS_APP_HELP_TEXT_HPP
#define SIRIUS_APP_HELP_TEXT_HPP

// The Help tab: what the reconstruction does and what every parameter group
// controls, as Qt rich text (HTML subset). Sections carry anchors the
// parameter panel's "?" buttons jump to: overview, pipeline, optics,
// sampling, filtering, run, viewer.

#include <QString>

namespace sirius::app {
    QString helpHtml();
}

#endif // SIRIUS_APP_HELP_TEXT_HPP
