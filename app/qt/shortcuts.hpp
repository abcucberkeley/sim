#ifndef SIRIUS_APP_SHORTCUTS_HPP
#define SIRIUS_APP_SHORTCUTS_HPP

// The shortcuts a panel or the viewer has to name in a tool tip, in one
// place, and the helper that renders them.
//
// The menu bar owns the actions, but the ops rows, the parameters footer, the
// viewer toolbar and the assistant button all advertise the same keys. They used to spell them
// out as Mac glyphs ("Move up (⌥↑)"), which is a lie on Windows and Linux;
// QKeySequence::NativeText prints what this platform actually uses ("Alt+Up",
// "⌥↑" on macOS), and taking the sequence from here keeps the tool tip and
// the action that fires from drifting apart.

#include <QKeySequence>
#include <QString>
#include <QStringList>

namespace sirius::app {

    namespace keys {
        inline QKeySequence runAll() { return QKeySequence(Qt::CTRL | Qt::Key_R); }
        inline QKeySequence runSelected() { return QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R); }
        inline QKeySequence removeStep() { return QKeySequence(Qt::Key_Backspace); }
        inline QKeySequence duplicateStep() { return QKeySequence(Qt::CTRL | Qt::Key_D); }
        inline QKeySequence enableStep() { return QKeySequence(Qt::Key_Space); }
        inline QKeySequence moveUp() { return QKeySequence(Qt::ALT | Qt::Key_Up); }
        inline QKeySequence moveDown() { return QKeySequence(Qt::ALT | Qt::Key_Down); }
        inline QKeySequence undo() { return QKeySequence(QKeySequence::Undo); }
        inline QKeySequence helpForStep() { return QKeySequence(Qt::Key_F1); }
        inline QKeySequence assistant() { return QKeySequence(Qt::ALT | Qt::Key_5); }
        inline QKeySequence logDock() { return QKeySequence(Qt::ALT | Qt::Key_7); }
        inline QKeySequence send() { return QKeySequence(Qt::Key_Return); }
    } // namespace keys

    // "Move step up" + Alt+Up -> "Move step up (Alt+Up)".
    inline QString shortcutText(const QKeySequence& key) { return key.toString(QKeySequence::NativeText); }

    inline QString withShortcut(const QString& text, const QKeySequence& key) {
        const QString s = shortcutText(key);
        return s.isEmpty() ? text : text + QStringLiteral(" (") + s + QStringLiteral(")");
    }

} // namespace sirius::app

#endif // SIRIUS_APP_SHORTCUTS_HPP
