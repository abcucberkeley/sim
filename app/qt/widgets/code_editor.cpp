#include "qt/widgets/code_editor.hpp"

#include <algorithm>

#include <QEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QTextBlock>

#include "qt/theme.hpp"

namespace sirius::app::widgets {

    // --- highlighter -------------------------------------------------------------

    PythonHighlighter::PythonHighlighter(QTextDocument* document) : QSyntaxHighlighter(document) {
        auto fmt = [](const QColor& c, bool bold = false, bool italic = false) {
            QTextCharFormat f;
            f.setForeground(c);
            if (bold) f.setFontWeight(QFont::Bold);
            if (italic) f.setFontItalic(true);
            return f;
        };
        const QTextCharFormat keyword = fmt(theme::kAccent700, true);
        const QTextCharFormat builtin = fmt(theme::kNeutral800);
        const QTextCharFormat number = fmt(theme::kAccent600);
        const QTextCharFormat decorator = fmt(theme::kNeutral600, false, true);
        const QTextCharFormat entry = fmt(theme::kAccent, true);   // STEP / run: the plugin contract
        const QTextCharFormat definition = fmt(theme::kText, true);
        string_ = fmt(theme::kNeutral700);
        comment_ = fmt(theme::kNeutral500, false, true);

        static const char* const keywords[] = {
            "and",  "as",   "assert", "async",  "await",   "break", "class",    "continue", "def",    "del",
            "elif", "else", "except", "False",  "finally", "for",   "from",     "global",   "if",     "import",
            "in",   "is",   "lambda", "None",   "nonlocal", "not",  "or",       "pass",     "raise",  "return",
            "True", "try",  "while",  "with",   "yield"};
        for (const char* k : keywords)
            rules_.push_back({QRegularExpression(QStringLiteral("\\b%1\\b").arg(QLatin1String(k))), keyword});
        static const char* const builtins[] = {
            "abs", "all",   "any", "dict",   "enumerate", "float", "int",   "isinstance", "len", "list",  "map", "max",
            "min", "print", "range", "round", "set",      "sorted", "str", "sum",        "tuple", "zip", "np",  "numpy"};
        for (const char* b : builtins)
            rules_.push_back({QRegularExpression(QStringLiteral("\\b%1\\b").arg(QLatin1String(b))), builtin});
        rules_.push_back({QRegularExpression(QStringLiteral("\\b[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?\\b")), number});
        rules_.push_back({QRegularExpression(QStringLiteral("^\\s*@\\w[\\w.]*")), decorator});
        rules_.push_back({QRegularExpression(QStringLiteral("\\bdef\\s+(\\w+)")), definition});
        rules_.push_back({QRegularExpression(QStringLiteral("\\bclass\\s+(\\w+)")), definition});
        rules_.push_back({QRegularExpression(QStringLiteral("\\bSTEP\\b")), entry});
        rules_.push_back({QRegularExpression(QStringLiteral("\\bdef\\s+(run)\\b")), entry});
        tripleDouble_ = QRegularExpression(QStringLiteral("\"\"\""));
        tripleSingle_ = QRegularExpression(QStringLiteral("'''"));
    }

    void PythonHighlighter::highlightBlock(const QString& text) {
        for (const Rule& r : rules_) {
            QRegularExpressionMatchIterator it = r.pattern.globalMatch(text);
            while (it.hasNext()) {
                const QRegularExpressionMatch m = it.next();
                // capture group 1 (the name after def / class) when present
                const int group = m.lastCapturedIndex() >= 1 ? 1 : 0;
                setFormat(static_cast<int>(m.capturedStart(group)), static_cast<int>(m.capturedLength(group)), r.format);
            }
        }
        // Single-line strings and comments, left to right so a '#' inside a
        // string stays a string and quotes inside a comment stay a comment.
        const int n = static_cast<int>(text.size());
        int i = 0;
        while (i < n) {
            const QChar c = text.at(i);
            if (c == QLatin1Char('#')) {
                setFormat(i, n - i, comment_);
                break;
            }
            if (c == QLatin1Char('"') || c == QLatin1Char('\'')) {
                if (i + 2 < n && text.at(i + 1) == c && text.at(i + 2) == c) {   // triple quotes: handled below
                    i += 3;
                    continue;
                }
                int j = i + 1;
                while (j < n && text.at(j) != c) {
                    if (text.at(j) == QLatin1Char('\\')) ++j;
                    ++j;
                }
                setFormat(i, std::min(j + 1, n) - i, string_);
                i = j + 1;
                continue;
            }
            ++i;
        }
        // Triple-quoted strings across blocks: state 1 = inside """, 2 = inside '''.
        int state = previousBlockState() > 0 ? previousBlockState() : 0;
        int start = 0;
        auto nextOpen = [&](int from) {
            const int d = static_cast<int>(text.indexOf(tripleDouble_, from));
            const int s = static_cast<int>(text.indexOf(tripleSingle_, from));
            if (d >= 0 && (s < 0 || d <= s)) { state = 1; start = d; }
            else if (s >= 0) { state = 2; start = s; }
            else { state = 0; start = -1; }
        };
        if (state == 0) nextOpen(0);
        while (start >= 0) {
            const QRegularExpression& close = state == 1 ? tripleDouble_ : tripleSingle_;
            const int end = static_cast<int>(text.indexOf(close, start + 3));
            if (end < 0) {
                setFormat(start, n - start, string_);
                setCurrentBlockState(state);
                return;
            }
            setFormat(start, end - start + 3, string_);
            nextOpen(end + 3);
        }
        setCurrentBlockState(0);
    }

    // --- editor ------------------------------------------------------------------

    namespace {
        class Gutter : public QWidget {
        public:
            explicit Gutter(CodeEditor* editor) : QWidget(editor), editor_(editor) {}
            QSize sizeHint() const override { return {editor_->gutterWidth(), 0}; }

        protected:
            void paintEvent(QPaintEvent* e) override { editor_->paintGutter(e); }

        private:
            CodeEditor* editor_;
        };
    } // namespace

    CodeEditor::CodeEditor(QWidget* parent) : QPlainTextEdit(parent) {
        gutter_ = new Gutter(this);
        setFont(theme::mono(13));
        // The application style sheet gives every widget the body face; a
        // rule on the widget itself is what outranks it.
        setStyleSheet(QStringLiteral("QPlainTextEdit { font-family: monospace; font-size: 13px; background: %1; color: %2;"
                                     " border: none; padding: 0; selection-background-color: %3; selection-color: %1; }")
                          .arg(theme::hex(theme::kBg), theme::hex(theme::kText), theme::hex(theme::kAccent)));
        setLineWrapMode(QPlainTextEdit::NoWrap);
        setTabStopDistance(4 * fontMetrics().horizontalAdvance(QLatin1Char(' ')));
        document()->setDocumentMargin(6);
        setFrameShape(QFrame::NoFrame);
        highlighter_ = new PythonHighlighter(document());
        connect(this, &QPlainTextEdit::blockCountChanged, this, [this](int) { updateGutterWidth(); });
        connect(this, &QPlainTextEdit::updateRequest, this, &CodeEditor::updateGutter);
        connect(this, &QPlainTextEdit::cursorPositionChanged, this, &CodeEditor::highlightCurrentLine);
        updateGutterWidth();
        highlightCurrentLine();
    }

    int CodeEditor::gutterWidth() const {
        int digits = 1;
        for (int n = std::max(1, blockCount()); n >= 10; n /= 10) ++digits;
        return 14 + fontMetrics().horizontalAdvance(QLatin1Char('9')) * std::max(digits, 2);
    }

    void CodeEditor::updateGutterWidth() { setViewportMargins(gutterWidth(), 0, 0, 0); }

    void CodeEditor::updateGutter(const QRect& rect, int dy) {
        if (dy) gutter_->scroll(0, dy);
        else gutter_->update(0, rect.y(), gutter_->width(), rect.height());
        if (rect.contains(viewport()->rect())) updateGutterWidth();
    }

    void CodeEditor::resizeEvent(QResizeEvent* event) {
        QPlainTextEdit::resizeEvent(event);
        const QRect cr = contentsRect();
        gutter_->setGeometry(QRect(cr.left(), cr.top(), gutterWidth(), cr.height()));
    }

    void CodeEditor::changeEvent(QEvent* event) {
        QPlainTextEdit::changeEvent(event);
        if (!gutter_) return;   // events raised while the base class is still being set up
        if (event->type() == QEvent::FontChange || event->type() == QEvent::StyleChange) {
            setTabStopDistance(4 * fontMetrics().horizontalAdvance(QLatin1Char(' ')));
            updateGutterWidth();
            const QRect cr = contentsRect();
            gutter_->setGeometry(QRect(cr.left(), cr.top(), gutterWidth(), cr.height()));
        } else if (event->type() == QEvent::ReadOnlyChange) {
            highlightCurrentLine();
        }
    }

    void CodeEditor::paintGutter(QPaintEvent* event) {
        QPainter p(gutter_);
        p.fillRect(event->rect(), theme::kSurface);
        p.setFont(font());
        QTextBlock block = firstVisibleBlock();
        int number = block.blockNumber();
        int top = static_cast<int>(blockBoundingGeometry(block).translated(contentOffset()).top());
        int bottom = top + static_cast<int>(blockBoundingRect(block).height());
        const int current = textCursor().blockNumber();
        while (block.isValid() && top <= event->rect().bottom()) {
            if (block.isVisible() && bottom >= event->rect().top()) {
                p.setPen(number == current ? theme::kText : theme::kNeutral500);
                p.drawText(0, top, gutter_->width() - 6, fontMetrics().height(), Qt::AlignRight,
                           QString::number(number + 1));
            }
            block = block.next();
            top = bottom;
            bottom = top + static_cast<int>(blockBoundingRect(block).height());
            ++number;
        }
    }

    void CodeEditor::highlightCurrentLine() {
        QList<QTextEdit::ExtraSelection> extra;
        if (!isReadOnly()) {
            QTextEdit::ExtraSelection sel;
            sel.format.setBackground(theme::kSurface);
            sel.format.setProperty(QTextFormat::FullWidthSelection, true);
            sel.cursor = textCursor();
            sel.cursor.clearSelection();
            extra.append(sel);
        }
        setExtraSelections(extra);
        gutter_->update();
    }

    void CodeEditor::toggleComment() {
        QTextCursor cursor = textCursor();
        const int from = cursor.selectionStart();
        const int to = cursor.selectionEnd();
        const QTextBlock first = document()->findBlock(from);
        const QTextBlock last = document()->findBlock(to);
        // Uncomment when every non-blank selected line is a comment, comment otherwise.
        bool allComments = true;
        for (QTextBlock b = first; b.isValid() && b.blockNumber() <= last.blockNumber(); b = b.next()) {
            const QString t = b.text().trimmed();
            if (!t.isEmpty() && !t.startsWith(QLatin1Char('#'))) allComments = false;
        }
        cursor.beginEditBlock();
        for (QTextBlock b = first; b.isValid() && b.blockNumber() <= last.blockNumber(); b = b.next()) {
            const QString text = b.text();
            int indent = 0;
            while (indent < text.size() && (text.at(indent) == QLatin1Char(' ') || text.at(indent) == QLatin1Char('\t')))
                ++indent;
            QTextCursor c(b);
            if (allComments) {
                const int hash = static_cast<int>(text.indexOf(QLatin1Char('#')));
                if (hash < 0) continue;
                const int len = text.mid(hash).startsWith(QLatin1String("# ")) ? 2 : 1;
                c.setPosition(b.position() + hash);
                c.setPosition(b.position() + hash + len, QTextCursor::KeepAnchor);
                c.removeSelectedText();
            } else if (!text.trimmed().isEmpty()) {
                c.setPosition(b.position() + indent);
                c.insertText(QStringLiteral("# "));
            }
        }
        cursor.endEditBlock();
    }

    void CodeEditor::keyPressEvent(QKeyEvent* event) {
        if (isReadOnly()) {
            QPlainTextEdit::keyPressEvent(event);
            return;
        }
        if (event->key() == Qt::Key_Tab && !(event->modifiers() & Qt::ControlModifier)) {
            insertPlainText(QStringLiteral("    "));
            return;
        }
        if (event->key() == Qt::Key_Slash && (event->modifiers() & Qt::ControlModifier)) {
            toggleComment();
            return;
        }
        if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
            const QTextCursor cur = textCursor();
            const QString line = cur.block().text().left(cur.positionInBlock());
            QString indent;
            for (QChar c : line) {
                if (c == QLatin1Char(' ') || c == QLatin1Char('\t')) indent += c;
                else break;
            }
            if (line.trimmed().endsWith(QLatin1Char(':'))) indent += QStringLiteral("    ");
            QPlainTextEdit::keyPressEvent(event);
            insertPlainText(indent);
            return;
        }
        QPlainTextEdit::keyPressEvent(event);
    }

} // namespace sirius::app::widgets
