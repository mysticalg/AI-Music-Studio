#pragma once

#include <juce_graphics/juce_graphics.h>

namespace aims::ui
{
constexpr float kTinyTextSize = 9.6f;
constexpr float kBodyTextSize = 10.4f;
constexpr float kStrongTextSize = 10.8f;
constexpr float kSectionTextSize = 11.2f;
constexpr float kTitleTextSize = 11.6f;

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
