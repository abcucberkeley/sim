#include "qt/viewer/dims_strip.hpp"

#include <algorithm>

#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>

#include "qt/theme.hpp"
#include "qt/viewer/viewer_widgets.hpp"

namespace sirius::app {

    namespace {
        QLabel* label(const QString& text, int px, int weight, const QColor& color, QWidget* parent) {
            auto* l = new QLabel(text, parent);
            QFont f(theme::kFontFamily);
            f.setPixelSize(px);
            f.setWeight(static_cast<QFont::Weight>(weight));
            l->setFont(f);
            QPalette pal = l->palette();
            pal.setColor(QPalette::WindowText, color);
            l->setPalette(pal);
            return l;
        }
        QSlider* slider(QWidget* parent) {
            auto* s = new QSlider(Qt::Horizontal, parent);
            s->setRange(0, 0);
            s->setSingleStep(1);
            s->setPageStep(1);
            s->setFocusPolicy(Qt::NoFocus);
            return s;
        }
    } // namespace

    DimsStrip::DimsStrip(QWidget* parent) : QWidget(parent) {
        auto* grid = new QGridLayout(this);
        grid->setContentsMargins(14, 8, 14, 8);
        grid->setHorizontalSpacing(14);
        grid->setVerticalSpacing(6);
        grid->setColumnMinimumWidth(0, 120);
        grid->setColumnStretch(1, 1);
        grid->setColumnMinimumWidth(2, 80);

        // Z row
        auto* zHead = new QWidget(this);
        auto* zh = new QHBoxLayout(zHead);
        zh->setContentsMargins(0, 0, 0, 0);
        zh->setSpacing(10);
        zh->addWidget(label(QStringLiteral("Z"), 12, QFont::ExtraBold, theme::kText, zHead));
        zUm_ = label(QStringLiteral("0.00 µm"), 12, QFont::Normal, theme::kNeutral600, zHead);
        zh->addWidget(zUm_);
        zh->addStretch(1);
        zSlider_ = slider(this);
        zPos_ = label(QStringLiteral("0 / 0"), 12, QFont::Normal, theme::kText, this);
        zPos_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        grid->addWidget(zHead, 0, 0);
        grid->addWidget(zSlider_, 0, 1);
        grid->addWidget(zPos_, 0, 2);

        // T row
        auto* tHead = new QWidget(this);
        auto* th = new QHBoxLayout(tHead);
        th->setContentsMargins(0, 0, 0, 0);
        th->setSpacing(10);
        th->addWidget(label(QStringLiteral("T"), 12, QFont::ExtraBold, theme::kText, tHead));
        play_ = new GlyphButton(QStringLiteral("▶"), tHead, QSize(20, 20));
        play_->setGlyphPx(9);
        play_->setBorderColor(theme::kText);
        play_->setToolTip(QStringLiteral("Play / pause the time series (space in the viewer)"));
        th->addWidget(play_);
        tSec_ = label(QStringLiteral("0.0 s"), 12, QFont::Normal, theme::kNeutral600, tHead);
        th->addWidget(tSec_);
        th->addStretch(1);
        tSlider_ = slider(this);
        tPos_ = label(QStringLiteral("0 / 0"), 12, QFont::Normal, theme::kText, this);
        tPos_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        grid->addWidget(tHead, 1, 0);
        grid->addWidget(tSlider_, 1, 1);
        grid->addWidget(tPos_, 1, 2);
        tRow_[0] = tHead;
        tRow_[1] = tSlider_;
        tRow_[2] = tPos_;

        connect(zSlider_, &QSlider::valueChanged, this, [this](int v) {
            if (!updating_) emit zRequested(static_cast<Index>(v));
        });
        connect(tSlider_, &QSlider::valueChanged, this, [this](int v) {
            if (!updating_) emit tRequested(static_cast<Index>(v));
        });
        connect(play_, &QAbstractButton::clicked, this, [this] { emit playToggled(!playing_); });
        setExtents(1, 1, 0.0, 0.0);
    }

    void DimsStrip::setExtents(Index nz, Index nt, double dzUm, double frameIntervalS) {
        nz_ = std::max<Index>(nz, 1);
        nt_ = std::max<Index>(nt, 1);
        dz_ = dzUm;
        dt_ = frameIntervalS;
        updating_ = true;
        zSlider_->setRange(0, static_cast<int>(nz_ - 1));
        tSlider_->setRange(0, static_cast<int>(nt_ - 1));
        zSlider_->setEnabled(nz_ > 1);
        updating_ = false;
        for (QWidget* w : tRow_) w->setVisible(nt_ > 1);
        z_ = std::clamp<Index>(z_, 0, nz_ - 1);
        t_ = std::clamp<Index>(t_, 0, nt_ - 1);
        setPosition(z_, t_);
    }

    void DimsStrip::setPosition(Index z, Index t) {
        z_ = std::clamp<Index>(z, 0, nz_ - 1);
        t_ = std::clamp<Index>(t, 0, nt_ - 1);
        updating_ = true;
        zSlider_->setValue(static_cast<int>(z_));
        tSlider_->setValue(static_cast<int>(t_));
        updating_ = false;
        refreshLabels();
    }

    void DimsStrip::setPlaying(bool on) {
        playing_ = on;
        play_->setGlyph(on ? QStringLiteral("❚❚") : QStringLiteral("▶"));
    }

    void DimsStrip::refreshLabels() {
        zUm_->setText(QString::number(static_cast<double>(z_) * dz_, 'f', 2) + QStringLiteral(" µm"));
        zPos_->setText(QStringLiteral("%1 / %2").arg(z_).arg(nz_ - 1));
        if (dt_ > 0.0) tSec_->setText(QString::number(static_cast<double>(t_) * dt_, 'f', 1) + QStringLiteral(" s"));
        else tSec_->setText(QStringLiteral("frame %1").arg(t_));
        tPos_->setText(QStringLiteral("%1 / %2").arg(t_).arg(nt_ - 1));
    }

} // namespace sirius::app
