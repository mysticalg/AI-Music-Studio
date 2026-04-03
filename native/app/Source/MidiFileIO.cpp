#include "MidiFileIO.h"

#include <limits>
#include <map>

namespace aims
{
namespace
{
struct ImportedTrackBuilder
{
    TrackState track;
    juce::String trackName;
    bool sawAnyEvent = false;
};

using BuilderKey = std::pair<int, int>;
using ActiveNoteKey = std::tuple<int, int, int>;

int scaleTick(double sourceTick, int sourceTicksPerQuarter)
{
    if (sourceTicksPerQuarter <= 0)
        return static_cast<int>(juce::roundToInt(sourceTick));

    return static_cast<int>(juce::roundToInt(sourceTick * (static_cast<double>(kTicksPerBeat) / static_cast<double>(sourceTicksPerQuarter))));
}

juce::String extractTrackName(const juce::MidiMessageSequence& sequence)
{
    for (int index = 0; index < sequence.getNumEvents(); ++index)
    {
        const auto* event = sequence.getEventPointer(index);
        if (event == nullptr)
            continue;

        const auto& message = event->message;
        if (message.isTrackNameEvent())
            return message.getTextFromTextMetaEvent().trim();
    }

    return {};
}

juce::String makeImportedTrackName(const ImportedTrackBuilder& builder, int sourceTrackIndex, int midiChannel)
{
    if (builder.trackName.isNotEmpty())
        return builder.trackName + " (Ch " + juce::String(midiChannel) + ")";

    return "Track " + juce::String(sourceTrackIndex + 1) + " (Ch " + juce::String(midiChannel) + ")";
}
} // namespace

juce::Result importMidiFileToProject(const juce::File& file, ProjectState& project)
{
    if (!file.existsAsFile())
        return juce::Result::fail("MIDI file not found.");

    juce::FileInputStream input(file);
    if (!input.openedOk())
        return juce::Result::fail("Failed to open MIDI file.");

    juce::MidiFile midi;
    if (!midi.readFrom(input, true))
        return juce::Result::fail("Failed to parse MIDI data.");

    const auto timeFormat = midi.getTimeFormat();
    if (timeFormat <= 0)
        return juce::Result::fail("SMPTE-timed MIDI files are not supported yet.");

    std::map<BuilderKey, ImportedTrackBuilder> builders;
    std::map<ActiveNoteKey, std::vector<std::pair<int, int>>> activeNotes;
    std::vector<TempoMarker> importedTempoMarkers;
    auto importedTimeSigNumerator = project.timeSigNumerator;
    auto importedTimeSigDenominator = project.timeSigDenominator;
    auto importedTimeSigTick = std::numeric_limits<int>::max();

    for (int trackIndex = 0; trackIndex < midi.getNumTracks(); ++trackIndex)
    {
        const auto* sequence = midi.getTrack(trackIndex);
        if (sequence == nullptr)
            continue;

        const auto trackName = extractTrackName(*sequence);

        for (int eventIndex = 0; eventIndex < sequence->getNumEvents(); ++eventIndex)
        {
            const auto* event = sequence->getEventPointer(eventIndex);
            if (event == nullptr)
                continue;

            const auto& message = event->message;
            const auto scaledTick = scaleTick(message.getTimeStamp(), timeFormat);

            if (message.isTempoMetaEvent())
            {
                const auto secondsPerQuarter = message.getTempoSecondsPerQuarterNote();
                if (secondsPerQuarter > 0.0)
                {
                    TempoMarker marker;
                    marker.tick = juce::jmax(0, scaledTick);
                    marker.bpm = juce::jlimit(20, 300, juce::roundToInt(60.0 / secondsPerQuarter));
                    importedTempoMarkers.push_back(marker);
                }
                continue;
            }

            if (message.isTimeSignatureMetaEvent())
            {
                int numerator = 4;
                int denominator = 4;
                message.getTimeSignatureInfo(numerator, denominator);
                if (scaledTick < importedTimeSigTick)
                {
                    importedTimeSigTick = scaledTick;
                    importedTimeSigNumerator = numerator;
                    importedTimeSigDenominator = denominator;
                }
                continue;
            }

            if (message.getChannel() <= 0)
                continue;

            const auto channel = juce::jlimit(1, 16, message.getChannel());
            auto& builder = builders[{ trackIndex, channel }];
            builder.trackName = trackName;
            builder.sawAnyEvent = true;
            builder.track.midiChannel = channel - 1;
            builder.track.instrumentMode = "General MIDI";
            builder.track.trackType = "instrument";

            if (message.isProgramChange())
            {
                builder.track.midiProgram = juce::jlimit(0, 127, message.getProgramChangeNumber());
                continue;
            }

            const auto noteNumber = juce::jlimit(0, 127, message.getNoteNumber());
            const auto noteKey = ActiveNoteKey(trackIndex, channel, noteNumber);

            if (message.isNoteOn())
            {
                const auto velocity = juce::jlimit(1, 127, static_cast<int>(message.getVelocity()));
                activeNotes[noteKey].emplace_back(scaledTick, velocity);
                continue;
            }

            if (message.isNoteOff())
            {
                auto activeIt = activeNotes.find(noteKey);
                if (activeIt == activeNotes.end() || activeIt->second.empty())
                    continue;

                const auto startInfo = activeIt->second.back();
                activeIt->second.pop_back();

                MidiNote note;
                note.startTick = juce::jmax(0, startInfo.first);
                note.durationTick = juce::jmax(1, scaledTick - startInfo.first);
                note.pitch = noteNumber;
                note.velocity = juce::jlimit(1, 127, startInfo.second);
                builder.track.notes.push_back(note);
            }
        }
    }

    std::vector<TrackState> importedTracks;
    importedTracks.reserve(builders.size());

    for (auto& [key, builder] : builders)
    {
        if (!builder.sawAnyEvent)
            continue;

        const auto sourceTrackIndex = key.first;
        const auto channel = key.second;
        builder.track.name = makeImportedTrackName(builder, sourceTrackIndex, channel);
        std::sort(builder.track.notes.begin(),
                  builder.track.notes.end(),
                  [] (const MidiNote& lhs, const MidiNote& rhs)
                  {
                      if (lhs.startTick != rhs.startTick)
                          return lhs.startTick < rhs.startTick;
                      if (lhs.pitch != rhs.pitch)
                          return lhs.pitch > rhs.pitch;
                      return lhs.durationTick < rhs.durationTick;
                  });
        importedTracks.push_back(std::move(builder.track));
    }

    if (importedTracks.empty())
    {
        TrackState track;
        track.name = "Track 1";
        importedTracks.push_back(std::move(track));
    }

    project.timeSigNumerator = normaliseTimeSignatureNumerator(importedTimeSigNumerator);
    project.timeSigDenominator = normaliseTimeSignatureDenominator(importedTimeSigDenominator);
    if (!importedTempoMarkers.empty())
    {
        sanitiseTempoMarkers(importedTempoMarkers, project.bpm);
        project.bpm = importedTempoMarkers.front().bpm;
        project.tempoMarkers = std::move(importedTempoMarkers);
    }
    else
    {
        project.tempoMarkers.clear();
    }

    project.tracks = std::move(importedTracks);
    project.midiPatterns.clear();
    project.midiSections.clear();
    project.playheadTick = 0;
    project.leftLocatorTick = 0;

    const auto projectBarTicks = ticksPerBar(project);
    int rightLocator = projectBarTicks;
    for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
    {
        auto& track = project.tracks[static_cast<size_t>(trackIndex)];
        for (const auto& note : track.notes)
            rightLocator = juce::jmax(rightLocator, note.startTick + note.durationTick);

        MidiPattern pattern;
        pattern.id = juce::Uuid().toString();
        pattern.name = track.name;
        pattern.notes = track.notes;

        int patternEndTick = projectBarTicks;
        for (const auto& note : pattern.notes)
            patternEndTick = juce::jmax(patternEndTick, note.startTick + note.durationTick);
        pattern.lengthTicks = juce::jmax(projectBarTicks, patternEndTick);
        project.midiPatterns.push_back(pattern);

        MidiSection section;
        section.trackIndex = trackIndex;
        section.startTick = 0;
        section.lengthTicks = pattern.lengthTicks;
        section.name = pattern.name;
        section.patternId = pattern.id;
        project.midiSections.push_back(section);
    }

    project.rightLocatorTick = juce::jmax(project.leftLocatorTick + 1,
                                          rightLocator + ticksPerTimeSignatureBeat(project));
    project.recalculateTimeFields();
    return juce::Result::ok();
}

juce::Result exportProjectToMidiFile(const juce::File& file, const ProjectState& project)
{
    if (file == juce::File())
        return juce::Result::fail("No destination file was provided.");

    juce::MidiFile midi;
    midi.setTicksPerQuarterNote(kTicksPerBeat);

    juce::MidiMessageSequence tempoTrack;
    auto tempoMarkers = project.tempoMarkers;
    if (tempoMarkers.empty())
        tempoMarkers.push_back({ 0, project.bpm });
    sanitiseTempoMarkers(tempoMarkers, project.bpm);

    tempoTrack.addEvent(juce::MidiMessage::timeSignatureMetaEvent(project.timeSigNumerator,
                                                                  project.timeSigDenominator).withTimeStamp(0.0));
    for (const auto& marker : tempoMarkers)
    {
        const auto tempoMicros = juce::jmax(1, 60000000 / juce::jmax(1, marker.bpm));
        tempoTrack.addEvent(juce::MidiMessage::tempoMetaEvent(tempoMicros).withTimeStamp(static_cast<double>(marker.tick)));
    }
    tempoTrack.addEvent(juce::MidiMessage::endOfTrack().withTimeStamp(static_cast<double>(project.rightLocatorTick)));
    tempoTrack.updateMatchedPairs();
    midi.addTrack(tempoTrack);

    for (const auto& track : project.tracks)
    {
        juce::MidiMessageSequence sequence;
        sequence.addEvent(juce::MidiMessage::textMetaEvent(0x03, track.name).withTimeStamp(0.0));

        if (track.instrumentMode.trim().equalsIgnoreCase("General MIDI"))
        {
            sequence.addEvent(juce::MidiMessage::programChange(track.midiChannel + 1,
                                                               juce::jlimit(0, 127, track.midiProgram)).withTimeStamp(0.0));
        }

        for (const auto& note : track.notes)
        {
            const auto startTick = static_cast<double>(juce::jmax(0, note.startTick));
            const auto endTick = static_cast<double>(juce::jmax(note.startTick + 1, note.startTick + note.durationTick));
            sequence.addEvent(juce::MidiMessage::noteOn(track.midiChannel + 1,
                                                        juce::jlimit(0, 127, note.pitch),
                                                        static_cast<juce::uint8>(juce::jlimit(1, 127, note.velocity))).withTimeStamp(startTick));
            sequence.addEvent(juce::MidiMessage::noteOff(track.midiChannel + 1,
                                                         juce::jlimit(0, 127, note.pitch)).withTimeStamp(endTick));
        }

        sequence.addEvent(juce::MidiMessage::endOfTrack().withTimeStamp(static_cast<double>(project.rightLocatorTick)));
        sequence.updateMatchedPairs();
        midi.addTrack(sequence);
    }

    const auto parent = file.getParentDirectory();
    if (!parent.createDirectory())
        return juce::Result::fail("Failed to create export folder: " + parent.getFullPathName());

    juce::FileOutputStream output(file);
    if (!output.openedOk())
        return juce::Result::fail("Failed to open destination MIDI file.");

    if (!midi.writeTo(output))
        return juce::Result::fail("Failed to write MIDI file.");

    return juce::Result::ok();
}

} // namespace aims
