#pragma once

#include <juce_graphics/juce_graphics.h>

namespace aims::ui
{
constexpr float kBodyTextSize = 10.4f;
constexpr float kTinyTextSize = kBodyTextSize;
constexpr float kStrongTextSize = kBodyTextSize;
constexpr float kSectionTextSize = kBodyTextSize;
constexpr float kTitleTextSize = kBodyTextSize;

inline juce::FontOptions font(float size = kBodyTextSize)
{
    return juce::FontOptions(size);
}

inline juce::FontOptions strongFont(float size = kStrongTextSize)
{
    return juce::FontOptions(size, juce::Font::bold);
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
    return juce::FontOptions(kTinyTextSize, bold ? juce::Font::bold : juce::Font::plain);
}
} // namespace aims::ui
