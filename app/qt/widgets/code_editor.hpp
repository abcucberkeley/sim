#ifndef SIRIUS_APP_WIDGETS_CODE_EDITOR_HPP
#define SIRIUS_APP_WIDGETS_CODE_EDITOR_HPP

// A small code editor for plugin files: QPlainTextEdit with a line-number
// gutter, the theme's monospace face, Python syntax highlighting and the
// few editing habits a quick edit needs (Tab = four spaces, auto-indent
// after ':', Ctrl+/ toggles comments). Not an IDE: no completion, no
// folding, no diagnostics.

#include <QPlainTextEdit>
#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QVector>

class QEvent;
class QPaintEvent;
class QResizeEvent;

namespace sirius::app::widgets {

    class PythonHighlighter : public QSyntaxHighlighter {
        Q_OBJECT
    public:
        explicit PythonHighlighter(QTextDocument* document);

    protected:
        void highlightBlock(const QString& text) override;

    private:
        struct Rule {
            QRegularExpression pattern;
            QTextCharFormat format;
        };
        QVector<Rule> rules_;
        QTextCharFormat string_;
        QTextCharFormat comment_;
        QRegularExpression tripleDouble_;
        QRegularExpression tripleSingle_;
    };

    class CodeEditor : public QPlainTextEdit {
        Q_OBJECT
    public:
        explicit CodeEditor(QWidget* parent = nullptr);

        int gutterWidth() const;
        void paintGutter(QPaintEvent* event);

    protected:
        void resizeEvent(QResizeEvent* event) override;
        void changeEvent(QEvent* event) override;
        void keyPressEvent(QKeyEvent* event) override;

    private:
        void updateGutterWidth();
        void updateGutter(const QRect& rect, int dy);
        void highlightCurrentLine();
        void toggleComment();

        QWidget* gutter_ = nullptr;
        PythonHighlighter* highlighter_ = nullptr;
    };

} // namespace sirius::app::widgets

#endif // SIRIUS_APP_WIDGETS_CODE_EDITOR_HPP
