#ifndef SIRIUS_APP_QT_STRINGS_HPP
#define SIRIUS_APP_QT_STRINGS_HPP

// QString <-> std::string without QString::toStdString/fromStdString.
//
// Those are inline in the Qt headers, but on MSVC a dllimport class may have
// its inline members taken from the DLL instead (Debug builds do), and the
// DLL's copy was compiled against a release std::string, whose layout differs
// from the debug one. A Debug sirius-app against a release-only Qt (the usual
// case: prebuilt Qt kits ship release DLLs) then corrupts every path it passes
// to the library. Going through QByteArray keeps all std::string code on our
// side of the boundary, for every Qt build.

#include <cstddef>
#include <string>

#include <QByteArray>
#include <QString>

namespace sirius::app {

    inline std::string toStd(const QString& s) {
        const QByteArray utf8 = s.toUtf8();
        return std::string(utf8.constData(), static_cast<std::size_t>(utf8.size()));
    }

    inline QString fromStd(const std::string& s) {
        return QString::fromUtf8(s.data(), static_cast<int>(s.size()));
    }

} // namespace sirius::app

#endif // SIRIUS_APP_QT_STRINGS_HPP
