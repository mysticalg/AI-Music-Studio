#include "MidiFileIO.h"

#include <array>
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

juce::String gmProgramName(int program)
{
    static const std::array<const char*, 128> names =
    {{
        "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
        "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
        "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
        "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
        "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
        "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
        "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
        "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
        "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
        "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
        "Violin", "Viola", "Cello", "Contrabass",
        "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
        "String Ensemble 1", "String Ensemble 2", "SynthStrings 1", "SynthStrings 2",
        "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
        "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
        "French Horn", "Brass Section", "SynthBrass 1", "SynthBrass 2",
        "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
        "Oboe", "English Horn", "Bassoon", "Clarinet",
        "Piccolo", "Flute", "Recorder", "Pan Flute",
        "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
        "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
        "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
        "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
        "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
        "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
        "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
        "Sitar", "Banjo", "Shamisen", "Koto",
        "Kalimba", "Bag pipe", "Fiddle", "Shanai",
        "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
        "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
        "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
        "Telephone Ring", "Helicopter", "Applause", "Gunshot"
    }};

    return names[static_cast<size_t>(juce::jlimit(0, 127, program))];
}

juce::String importedInstrumentName(int midiProgram, int midiChannel)
{
    if (juce::jlimit(0, 15, midiChannel) == 9)
        return "Standard Drum Kit";

    return gmProgramName(midiProgram);
}

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

juce::String trackExportIdentity(const ProjectState& project, const TrackState& track)
{
    juce::StringArray parts;
    parts.add(track.instrument.trim());
    parts.add(displayRackName(project, track).trim());
    parts.add(track.name.trim());
    parts.add(track.synthProfile.trim());
    return parts.joinIntoString(" ").toLowerCase();
}

bool trackExportsAsGmPercussion(const ProjectState& project, const TrackState& track)
{
    const auto identity = trackExportIdentity(project, track);
    const auto family = track.synthProfile.trim().toLowerCase();
    return track.midiChannel == 9
        || family == "noise_kit"
        || family == "drums"
        || identity.contains("drum")
        || identity.contains("kit")
        || identity.contains("808")
        || identity.contains("perc");
}

int gmProgramForTrackExport(const ProjectState& project, const TrackState& track)
{
    if (track.instrumentMode.trim().equalsIgnoreCase("General MIDI"))
        return juce::jlimit(0, 127, track.midiProgram);

    const auto identity = trackExportIdentity(project, track);
    const auto family = track.synthProfile.trim().toLowerCase();

    const std::pair<const char*, int> tokenMap[] =
    {
        { "bass guitar", 33 },
        { "fingered bass", 33 },
        { "bass synth", 38 },
        { "synth bass", 38 },
        { "tb303", 38 },
        { "violin", 40 },
        { "cello", 42 },
        { "strings", 48 },
        { "string", 48 },
        { "choir", 52 },
        { "voice", 54 },
        { "trumpet", 56 },
        { "trombone", 57 },
        { "horn", 60 },
        { "brass", 61 },
        { "sax", 65 },
        { "oboe", 68 },
        { "clarinet", 71 },
        { "flute", 73 },
        { "pipe", 73 },
        { "lead synth", 80 },
        { "lead", 80 },
        { "pad synth", 88 },
        { "pad", 88 },
        { "pluck", 24 },
        { "guitar", 24 },
        { "harp", 46 },
        { "organ", 19 },
        { "electric piano", 4 },
        { "ep", 4 },
        { "piano", 0 },
        { "bass", 33 },
    };

    for (const auto& [token, program] : tokenMap)
    {
        if (identity.contains(token))
            return program;
    }

    if (family == "sub_bass")
        return 38;
    if (family == "saw_pad")
        return 88;
    if (family == "brass_stack")
        return 61;
    if (family == "reed_breath")
        return identity.contains("flute") ? 73 : 65;
    if (family == "organ")
        return 19;
    if (family == "e_piano")
        return 4;
    if (family == "pluck")
        return 24;
    if (family == "synth")
        return 80;

    const auto groupedDefault = defaultMidiProgramForInstrumentName(identity);
    switch (groupedDefault)
    {
        case 8: return 11;
        case 16: return 19;
        case 24: return 24;
        case 32: return 33;
        case 40: return 48;
        case 48: return 52;
        case 56: return 61;
        case 64: return 65;
        case 72: return 73;
        case 80: return 80;
        case 88: return 88;
        case 96: return 96;
        case 104: return 104;
        case 112: return 112;
        case 120: return 120;
        default: break;
    }

    if (track.midiProgram > 0)
        return juce::jlimit(0, 127, track.midiProgram);

    return 0;
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
        builder.track.midiProgram = juce::jlimit(0, 127, builder.track.midiProgram);
        builder.track.instrument = importedInstrumentName(builder.track.midiProgram, builder.track.midiChannel);
        builder.track.synthProfile = inferSynthProfile(builder.track.instrument, builder.track.midiProgram);
        if (builder.track.midiChannel == 9)
            builder.track.synthProfile = "noise_kit";
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
        const auto exportsAsPercussion = trackExportsAsGmPercussion(project, track);
        const auto exportChannel = exportsAsPercussion ? 9 : juce::jlimit(0, 15, track.midiChannel);
        if (!exportsAsPercussion)
            sequence.addEvent(juce::MidiMessage::programChange(exportChannel + 1,
                                                               gmProgramForTrackExport(project, track)).withTimeStamp(0.0));

        for (const auto& note : track.notes)
        {
            const auto startTick = static_cast<double>(juce::jmax(0, note.startTick));
            const auto endTick = static_cast<double>(juce::jmax(note.startTick + 1, note.startTick + note.durationTick));
            sequence.addEvent(juce::MidiMessage::noteOn(exportChannel + 1,
                                                        juce::jlimit(0, 127, note.pitch),
                                                        static_cast<juce::uint8>(juce::jlimit(1, 127, note.velocity))).withTimeStamp(startTick));
            sequence.addEvent(juce::MidiMessage::noteOff(exportChannel + 1,
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
