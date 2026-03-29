#pragma once

#include "ProjectModel.h"

#include <juce_audio_basics/juce_audio_basics.h>

namespace aims
{

juce::Result importMidiFileToProject(const juce::File& file, ProjectState& project);
juce::Result exportProjectToMidiFile(const juce::File& file, const ProjectState& project);

} // namespace aims
