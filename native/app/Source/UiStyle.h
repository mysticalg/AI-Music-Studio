#pragma once

#include <juce_graphics/juce_graphics.h>

namespace aims::ui
{
constexpr float kBodyTextSize = 10.4f;
constexpr float kTinyTextSize = kBodyTextSize;
constexpr float kStrongTextSize = kBodyTextSize;
constexpr float kSectionTextSize = kBodyTextSize;
constexpr float kTitleTextSize = kBodyTextSize;

inline float& fontScaleStorage() noexcept
{
    static float scale = 1.0f;
    return scale;
}

inline float fontScale() noexcept
{
    return fontScaleStorage();
}

inline void setFontScale(float scale) noexcept
{
    fontScaleStorage() = juce::jlimit(0.5f, 2.0f, scale);
}

inline float scaleValue(float value) noexcept
{
    return value * fontScale();
}

inline juce::FontOptions font(float size = kBodyTextSize)
{
    return juce::FontOptions(scaleValue(size));
}

inline juce::FontOptions strongFont(float size = kStrongTextSize)
{
    return juce::FontOptions(scaleValue(size), juce::Font::bold);
}

inline juce::FontOptions sectionFont()
{
    return strongFont(kSectionTextSize);
}

inline juce::FontOptions titleFont()
{
    return strongFont(kTitleTextSize);
}

inline juce::FontOptions tinyFont(bool bold = false)
{
    return juce::FontOptions(scaleValue(kTinyTextSize), bold ? juce::Font::bold : juce::Font::plain);
}
} // namespace aims::ui
