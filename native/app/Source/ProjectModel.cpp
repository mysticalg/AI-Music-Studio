#include "ProjectModel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace aims
{
namespace
{
struct TrackColourThemeState
{
    juce::Colour accent = juce::Colour::fromRGB(72, 104, 160);
    juce::Colour surface = juce::Colour::fromRGB(20, 22, 28);
    juce::Colour outline = juce::Colour::fromRGB(56, 64, 79);
    juce::Colour success = juce::Colour::fromRGB(143, 225, 170);
    juce::Colour info = juce::Colour::fromRGB(156, 199, 239);
    juce::Colour warning = juce::Colour::fromRGB(230, 190, 118);
};

TrackColourThemeState& trackColourThemeState()
{
    static TrackColourThemeState state;
    return state;
}

juce::Colour rotateTrackHue(const juce::Colour& colour, float delta)
{
    auto hue = colour.getHue() + delta;
    hue -= std::floor(hue);
    if (hue < 0.0f)
        hue += 1.0f;

    const auto saturation = juce::jlimit(0.42f, 0.9f, (colour.getSaturation() * 0.94f) + 0.03f);
    const auto brightness = juce::jlimit(0.48f, 0.94f, (colour.getBrightness() * 0.92f) + 0.04f);
    return juce::Colour::fromHSV(hue, saturation, brightness, 1.0f);
}

juce::Colour normaliseThemeSeed(const juce::Colour& seed)
{
    const auto& theme = trackColourThemeState();
    auto colour = seed.interpolatedWith(theme.surface.contrasting(0.7f), 0.08f);
    colour = colour.interpolatedWith(theme.outline, 0.1f);

    const auto saturation = juce::jlimit(0.44f, 0.9f, (colour.getSaturation() * 0.92f) + 0.05f);
    const auto brightness = juce::jlimit(0.5f, 0.94f, (colour.getBrightness() * 0.9f) + 0.05f);
    return juce::Colour::fromHSV(colour.getHue(), saturation, brightness, 1.0f);
}

std::array<juce::Colour, 8> themedTrackPalette()
{
    const auto& theme = trackColourThemeState();

    const auto accent = normaliseThemeSeed(theme.accent);
    const auto success = normaliseThemeSeed(theme.success);
    const auto info = normaliseThemeSeed(theme.info);
    const auto warning = normaliseThemeSeed(theme.warning);

    return { {
        accent,
        normaliseThemeSeed(warning.interpolatedWith(accent, 0.18f)),
        success,
        info,
        normaliseThemeSeed(rotateTrackHue(accent, 0.12f)),
        normaliseThemeSeed(rotateTrackHue(warning.interpolatedWith(accent, 0.34f), -0.08f)),
        normaliseThemeSeed(success.interpolatedWith(info, 0.46f)),
        normaliseThemeSeed(rotateTrackHue(info.interpolatedWith(warning, 0.3f), -0.13f))
    } };
}

template <typename Value>
Value clampValue(Value value, Value minimum, Value maximum)
{
    return juce::jlimit(minimum, maximum, value);
}

int nearestTimeSignatureDenominator(int denominator)
{
    static constexpr int allowedDenominators[] { 1, 2, 4, 8, 16, 32 };

    auto best = 4;
    auto bestDistance = std::numeric_limits<int>::max();
    for (const auto allowed : allowedDenominators)
    {
        const auto distance = std::abs(denominator - allowed);
        if (distance < bestDistance)
        {
            best = allowed;
            bestDistance = distance;
        }
    }

    return best;
}

struct KeyQuantizeScaleDefinition
{
    const char* id;
    const char* label;
    std::initializer_list<int> intervals;
};

struct MidiControllerDefinition
{
    int controller = 0;
    const char* label = "";
};

const std::array<KeyQuantizeScaleDefinition, 10>& keyQuantizeScaleDefinitions()
{
    static const std::array<KeyQuantizeScaleDefinition, 10> definitions { {
        { "chromatic", "Chromatic", { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 } },
        { "major", "Major", { 0, 2, 4, 5, 7, 9, 11 } },
        { "minor", "Minor", { 0, 2, 3, 5, 7, 8, 10 } },
        { "pentatonic_major", "Pentatonic Major", { 0, 2, 4, 7, 9 } },
        { "pentatonic_minor", "Pentatonic Minor", { 0, 3, 5, 7, 10 } },
        { "harmonic_minor", "Harmonic Minor", { 0, 2, 3, 5, 7, 8, 11 } },
        { "melodic_minor", "Melodic Minor", { 0, 2, 3, 5, 7, 9, 11 } },
        { "diminished_whole_half", "Diminished (Whole-Half)", { 0, 2, 3, 5, 6, 8, 9, 11 } },
        { "diminished_half_whole", "Diminished (Half-Whole)", { 0, 1, 3, 4, 6, 7, 9, 10 } },
        { "augmented", "Augmented", { 0, 3, 4, 7, 8, 11 } }
    } };

    return definitions;
}

const KeyQuantizeScaleDefinition* findKeyQuantizeScaleDefinition(const juce::String& scaleId)
{
    const auto trimmed = scaleId.trim();
    for (const auto& definition : keyQuantizeScaleDefinitions())
    {
        if (trimmed.equalsIgnoreCase(definition.id))
            return &definition;
    }

    return nullptr;
}

const std::array<MidiControllerDefinition, 9>& midiControllerDefinitions()
{
    static const std::array<MidiControllerDefinition, 9> definitions { {
        { 1, "Mod Wheel" },
        { 7, "Channel Volume" },
        { 10, "Pan" },
        { 11, "Expression" },
        { 64, "Sustain" },
        { 71, "Resonance" },
        { 74, "Brightness" },
        { 91, "Reverb Send" },
        { 93, "Chorus Send" }
    } };

    return definitions;
}

const MidiControllerDefinition* findMidiControllerDefinition(int controllerNumber)
{
    for (const auto& definition : midiControllerDefinitions())
    {
        if (definition.controller == controllerNumber)
            return &definition;
    }

    return nullptr;
}

juce::var serialiseAutomationPoint(const AutomationPoint& point);
juce::var serialiseAutomationLane(const AutomationLane& lane);

juce::var makeObjectVar()
{
    return juce::var(new juce::DynamicObject());
}

void setObjectProperty(juce::var& objectVar, const juce::Identifier& name, const juce::var& value)
{
    if (auto* object = objectVar.getDynamicObject())
        object->setProperty(name, value);
}

juce::var getObjectProperty(const juce::var& objectVar, const juce::Identifier& name, const juce::var& fallback = {})
{
    if (auto* object = objectVar.getDynamicObject())
        return object->getProperty(name);
    return fallback;
}

juce::String getStringProperty(const juce::var& objectVar, const juce::Identifier& name, const juce::String& fallback = {})
{
    const auto value = getObjectProperty(objectVar, name);
    return value.isVoid() ? fallback : value.toString();
}

int getIntProperty(const juce::var& objectVar,
                   const juce::Identifier& name,
                   int fallback,
                   int minimum = std::numeric_limits<int>::lowest(),
                   int maximum = std::numeric_limits<int>::max())
{
    const auto value = getObjectProperty(objectVar, name);
    if (value.isVoid())
        return fallback;
    return clampValue(static_cast<int>(value), minimum, maximum);
}

double getDoubleProperty(const juce::var& objectVar,
                         const juce::Identifier& name,
                         double fallback,
                         double minimum = -std::numeric_limits<double>::max(),
                         double maximum = std::numeric_limits<double>::max())
{
    const auto value = getObjectProperty(objectVar, name);
    if (value.isVoid())
        return fallback;

    const auto parsed = static_cast<double>(value);
    if (!std::isfinite(parsed))
        return fallback;
    return clampValue(parsed, minimum, maximum);
}

bool getBoolProperty(const juce::var& objectVar, const juce::Identifier& name, bool fallback)
{
    const auto value = getObjectProperty(objectVar, name);
    return value.isVoid() ? fallback : static_cast<bool>(value);
}

juce::StringArray readStringArray(const juce::var& value)
{
    juce::StringArray result;
    if (auto* array = value.getArray())
    {
        for (const auto& item : *array)
            result.add(item.toString());
    }
    return result;
}

juce::Array<juce::var> readVarArray(const juce::var& value)
{
    juce::Array<juce::var> result;
    if (auto* array = value.getArray())
    {
        for (const auto& item : *array)
            result.add(item);
    }
    return result;
}

std::vector<bool> readBoolVector(const juce::var& value)
{
    std::vector<bool> result;
    if (auto* array = value.getArray())
    {
        result.reserve(static_cast<size_t>(array->size()));
        for (const auto& item : *array)
            result.push_back(static_cast<bool>(item));
    }
    return result;
}

std::vector<float> readFloatVector(const juce::var& value)
{
    std::vector<float> result;
    if (auto* array = value.getArray())
    {
        result.reserve(static_cast<size_t>(array->size()));
        for (const auto& item : *array)
        {
            const auto parsed = static_cast<double>(item);
            if (std::isfinite(parsed))
                result.push_back(static_cast<float>(clampValue(parsed, -1.0, 1.0)));
        }
    }
    return result;
}

juce::var makeStringArrayVar(const juce::StringArray& values)
{
    juce::Array<juce::var> array;
    for (const auto& value : values)
        array.add(value);
    return juce::var(array);
}

juce::var makeVarArray(const juce::Array<juce::var>& values)
{
    juce::Array<juce::var> array;
    for (const auto& value : values)
        array.add(value);
    return juce::var(array);
}

juce::var makeBoolArrayVar(const std::vector<bool>& values, int expectedSize = -1)
{
    juce::Array<juce::var> array;
    const auto count = expectedSize >= 0 ? expectedSize : static_cast<int>(values.size());
    for (int index = 0; index < count; ++index)
    {
        const auto enabled = index >= 0
            && index < static_cast<int>(values.size())
            && values[static_cast<size_t>(index)];
        array.add(enabled);
    }
    return juce::var(array);
}

juce::var makeFloatArrayVar(const std::vector<float>& values)
{
    juce::Array<juce::var> array;
    for (const auto value : values)
        array.add(static_cast<double>(value));
    return juce::var(array);
}

std::vector<bool> normaliseBoolVector(const std::vector<bool>& values, int expectedSize)
{
    std::vector<bool> result(static_cast<size_t>(juce::jmax(0, expectedSize)), false);
    const auto count = juce::jmin(static_cast<int>(values.size()), expectedSize);
    for (int index = 0; index < count; ++index)
        result[static_cast<size_t>(index)] = values[static_cast<size_t>(index)];
    return result;
}

juce::var serialiseMidiNote(const MidiNote& note)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "start_tick", note.startTick);
    setObjectProperty(value, "duration_tick", note.durationTick);
    setObjectProperty(value, "pitch", note.pitch);
    setObjectProperty(value, "velocity", note.velocity);
    setObjectProperty(value, "selected", note.selected);
    return value;
}

juce::var serialiseMidiPattern(const MidiPattern& pattern, int barTicks = kTicksPerBar)
{
    auto value = makeObjectVar();
    juce::Array<juce::var> notes;
    juce::Array<juce::var> controllerLanes;
    for (const auto& note : pattern.notes)
        notes.add(serialiseMidiNote(note));
    for (const auto& lane : pattern.controllerLanes)
        controllerLanes.add(serialiseAutomationLane(lane));

    setObjectProperty(value, "id", pattern.id);
    setObjectProperty(value, "name", pattern.name);
    setObjectProperty(value, "length_ticks", pattern.lengthTicks);
    setObjectProperty(value, "length_bars", juce::jmax(1, (pattern.lengthTicks + barTicks - 1) / barTicks));
    setObjectProperty(value, "notes", juce::var(notes));
    setObjectProperty(value, "controller_lanes", juce::var(controllerLanes));
    return value;
}

juce::var serialiseAutomationPoint(const AutomationPoint& point)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "tick", point.tick);
    setObjectProperty(value, "value", point.value);
    return value;
}

juce::var serialiseAutomationLane(const AutomationLane& lane)
{
    auto value = makeObjectVar();
    juce::Array<juce::var> points;
    for (const auto& point : lane.points)
        points.add(serialiseAutomationPoint(point));
    setObjectProperty(value, "target", lane.target);
    setObjectProperty(value, "enabled", lane.enabled);
    setObjectProperty(value, "points", juce::var(points));
    return value;
}

juce::var serialiseTempoMarker(const TempoMarker& marker)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "tick", marker.tick);
    setObjectProperty(value, "bpm", marker.bpm);
    return value;
}

juce::var serialiseTrack(const TrackState& track)
{
    auto value = makeObjectVar();
    juce::Array<juce::var> notes;
    juce::Array<juce::var> automationLanes;
    auto parametersVar = makeObjectVar();

    for (const auto& note : track.notes)
        notes.add(serialiseMidiNote(note));

    for (const auto& lane : track.automationLanes)
        automationLanes.add(serialiseAutomationLane(lane));

    for (int index = 0; index < track.vstiParameters.size(); ++index)
    {
        const auto key = track.vstiParameters.getName(index).toString();
        setObjectProperty(parametersVar, key, track.vstiParameters.getValueAt(index));
    }

    setObjectProperty(value, "name", track.name);
    setObjectProperty(value, "track_type", track.trackType);
    setObjectProperty(value, "notes", juce::var(notes));
    setObjectProperty(value, "volume", track.volume);
    setObjectProperty(value, "pan", track.pan);
    setObjectProperty(value, "instrument", track.instrument);
    setObjectProperty(value, "instrument_mode", track.instrumentMode);
    setObjectProperty(value, "rack_vsti", track.rackVst);
    setObjectProperty(value, "plugins", makeStringArrayVar(track.plugins));
    setObjectProperty(value, "vsti_parameters", parametersVar);
    setObjectProperty(value, "vsti_state_path", track.vstiStatePath);
    setObjectProperty(value, "vsti_state_b64", track.vstiStateBase64);
    setObjectProperty(value, "vsti_output_gain_db", track.vstiOutputGainDb);
    setObjectProperty(value, "vsti_wet_mix", track.vstiWetMix);
    setObjectProperty(value, "vsti_output_bus_routes", makeVarArray(track.vstiOutputBusRoutes));
    setObjectProperty(value, "vsti_input_bus_routes", makeVarArray(track.vstiInputBusRoutes));
    setObjectProperty(value, "vst_fx_chain", makeStringArrayVar(track.vstFxChain));
    setObjectProperty(value, "vst_fx_bypassed", track.vstFxBypassed);
    setObjectProperty(value, "vst_fx_slot_bypassed",
                      makeBoolArrayVar(track.vstFxSlotBypassed, track.vstFxChain.size()));
    setObjectProperty(value, "midi_program", track.midiProgram);
    setObjectProperty(value, "midi_channel", track.midiChannel);
    setObjectProperty(value, "synth_profile", track.synthProfile);
    setObjectProperty(value, "rendered_audio_path", track.renderedAudioPath);
    setObjectProperty(value, "mute", track.mute);
    setObjectProperty(value, "solo", track.solo);
    setObjectProperty(value, "live_armed", track.liveArmed);
    setObjectProperty(value, "color_hex", track.colorHex);
    setObjectProperty(value, "theme_track_colour", track.followThemeTrackColour);
    setObjectProperty(value, "theme_colour_slot", track.themeColourSlot);
    setObjectProperty(value, "automation_lanes", juce::var(automationLanes));
    setObjectProperty(value,
                      "arrangement_visible_automation_targets",
                      makeStringArrayVar(track.arrangementVisibleAutomationTargets));
    setObjectProperty(value, "routing_target", track.routingTarget);
    return value;
}

juce::var serialiseSharedEffectBus(const SharedEffectBusState& bus)
{
    auto value = makeObjectVar();
    auto parametersVar = makeObjectVar();
    for (int index = 0; index < bus.parameters.size(); ++index)
    {
        const auto key = bus.parameters.getName(index).toString();
        setObjectProperty(parametersVar, key, bus.parameters.getValueAt(index));
    }

    setObjectProperty(value, "id", bus.id);
    setObjectProperty(value, "name", bus.name);
    setObjectProperty(value, "effect", bus.effect);
    setObjectProperty(value, "output_targets", makeStringArrayVar(bus.outputTargets));
    setObjectProperty(value, "parameters", parametersVar);
    setObjectProperty(value, "state_path", bus.statePath);
    setObjectProperty(value, "bypassed", bus.bypassed);
    return value;
}

juce::var serialiseVstInstrument(const VstInstrument& instrument)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "name", instrument.name);
    setObjectProperty(value, "path", instrument.path);
    setObjectProperty(value, "plugin_name", instrument.pluginName);
    setObjectProperty(value, "is_instrument", instrument.isInstrument);
    setObjectProperty(value, "is_effect", instrument.isEffect);
    setObjectProperty(value, "category", instrument.category);
    setObjectProperty(value, "host_supported", instrument.hostSupported);
    setObjectProperty(value, "host_error", instrument.hostError);
    return value;
}

juce::var serialiseSampleAsset(const SampleAsset& asset)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "path", asset.path);
    setObjectProperty(value, "duration_sec", asset.durationSec);
    setObjectProperty(value, "sample_rate", asset.sampleRate);
    setObjectProperty(value, "waveform_preview", makeFloatArrayVar(asset.waveformPreview));
    return value;
}

juce::var serialiseSampleClip(const SampleClip& clip)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "path", clip.path);
    setObjectProperty(value, "track_index", clip.trackIndex);
    setObjectProperty(value, "start_sec", clip.startSec);
    setObjectProperty(value, "duration_sec", clip.durationSec);
    setObjectProperty(value, "source_offset_sec", clip.sourceOffsetSec);
    setObjectProperty(value, "source_file_duration_sec", clip.sourceFileDurationSec);
    setObjectProperty(value, "sample_rate", clip.sampleRate);
    setObjectProperty(value, "waveform_preview", makeFloatArrayVar(clip.waveformPreview));
    return value;
}

juce::var serialiseMidiSection(const MidiSection& section, int barTicks = kTicksPerBar)
{
    auto value = makeObjectVar();
    setObjectProperty(value, "track_index", section.trackIndex);
    setObjectProperty(value, "start_tick", section.startTick);
    setObjectProperty(value, "length_ticks", section.lengthTicks);
    setObjectProperty(value, "length_bars", juce::jmax(1, (section.lengthTicks + barTicks - 1) / barTicks));
    setObjectProperty(value, "start_sec", section.startSec);
    setObjectProperty(value, "duration_sec", section.durationSec);
    setObjectProperty(value, "name", section.name);
    setObjectProperty(value, "pattern_id", section.patternId);
    return value;
}

juce::var serialiseProjectState(const ProjectState& project)
{
    auto value = makeObjectVar();
    juce::Array<juce::var> trackArray;
    juce::Array<juce::var> rackArray;
    juce::Array<juce::var> assetArray;
    juce::Array<juce::var> clipArray;
    juce::Array<juce::var> patternArray;
    juce::Array<juce::var> sectionArray;
    juce::Array<juce::var> tempoMarkerArray;
    juce::Array<juce::var> sharedFxBusArray;
    const auto projectBarTicks = ticksPerBar(project);

    for (const auto& track : project.tracks)
        trackArray.add(serialiseTrack(track));

    for (const auto& instrument : project.vstRack)
        rackArray.add(serialiseVstInstrument(instrument));

    for (const auto& asset : project.sampleAssets)
        assetArray.add(serialiseSampleAsset(asset));

    for (const auto& clip : project.sampleClips)
        clipArray.add(serialiseSampleClip(clip));

    for (const auto& pattern : project.midiPatterns)
        patternArray.add(serialiseMidiPattern(pattern, projectBarTicks));

    for (const auto& section : project.midiSections)
        sectionArray.add(serialiseMidiSection(section, projectBarTicks));

    for (const auto& marker : project.tempoMarkers)
        tempoMarkerArray.add(serialiseTempoMarker(marker));

    for (const auto& bus : project.sharedFxBuses)
        sharedFxBusArray.add(serialiseSharedEffectBus(bus));

    setObjectProperty(value, "bpm", project.bpm);
    setObjectProperty(value, "time_signature_numerator", project.timeSigNumerator);
    setObjectProperty(value, "time_signature_denominator", project.timeSigDenominator);
    setObjectProperty(value, "default_pattern_ticks", project.defaultPatternTicks);
    setObjectProperty(value, "default_pattern_bars", juce::jmax(1, (project.defaultPatternTicks + projectBarTicks - 1) / projectBarTicks));
    setObjectProperty(value, "quantize_enabled", project.quantizeEnabled);
    setObjectProperty(value, "quantize_div", project.quantizeDiv);
    setObjectProperty(value, "quantize_triplet", project.quantizeTriplet);
    setObjectProperty(value, "key_quantize_root", project.keyQuantizeRoot);
    setObjectProperty(value, "key_quantize_scale", project.keyQuantizeScale);
    setObjectProperty(value, "piano_roll_visible_pitch_min", project.pianoRollVisiblePitchMin);
    setObjectProperty(value, "piano_roll_visible_pitch_max", project.pianoRollVisiblePitchMax);
    setObjectProperty(value, "arrangement_snap_ticks", project.arrangementSnapTicks);
    setObjectProperty(value, "loop_enabled", project.loopEnabled);
    setObjectProperty(value, "metronome_enabled", project.metronomeEnabled);
    setObjectProperty(value, "left_locator_tick", project.leftLocatorTick);
    setObjectProperty(value, "right_locator_tick", project.rightLocatorTick);
    setObjectProperty(value, "playhead_tick", project.playheadTick);
    setObjectProperty(value, "left_locator_sec", project.leftLocatorSec);
    setObjectProperty(value, "right_locator_sec", project.rightLocatorSec);
    setObjectProperty(value, "playhead_sec", project.playheadSec);
    setObjectProperty(value, "master_volume", project.masterVolume);
    setObjectProperty(value, "master_fx_chain", makeStringArrayVar(project.masterFxChain));
    setObjectProperty(value, "master_fx_bypassed", project.masterFxBypassed);
    setObjectProperty(value, "master_fx_slot_bypassed",
                      makeBoolArrayVar(project.masterFxSlotBypassed, project.masterFxChain.size()));
    setObjectProperty(value, "shared_fx_buses", juce::var(sharedFxBusArray));
    setObjectProperty(value, "tempo_markers", juce::var(tempoMarkerArray));
    setObjectProperty(value, "tracks", juce::var(trackArray));
    setObjectProperty(value, "vsti_paths", makeStringArrayVar(project.vstiPaths));
    setObjectProperty(value, "vsti_folder_paths", makeStringArrayVar(project.vstiFolderPaths));
    setObjectProperty(value, "sample_paths", makeStringArrayVar(project.samplePaths));
    setObjectProperty(value, "vsti_rack", juce::var(rackArray));
    setObjectProperty(value, "sample_assets", juce::var(assetArray));
    setObjectProperty(value, "sample_clips", juce::var(clipArray));
    setObjectProperty(value, "midi_patterns", juce::var(patternArray));
    setObjectProperty(value, "midi_sections", juce::var(sectionArray));
    return value;
}

TrackState makeDefaultTrack(const juce::String& name,
                            const juce::String& rackName,
                            int midiChannel,
                            const juce::String& synthProfile)
{
    TrackState track;
    track.name = name;
    track.instrument = rackName;
    track.instrumentMode = "VSTI Rack";
    track.rackVst = rackName;
    track.midiChannel = midiChannel;
    track.synthProfile = synthProfile;
    track.followThemeTrackColour = true;
    return track;
}

std::vector<TrackState> makeDefaultTracks()
{
    auto tracks = std::vector<TrackState> {
        makeDefaultTrack("Track 1", "AI Piano", 0, "keys"),
    };

    for (size_t index = 0; index < tracks.size(); ++index)
    {
        tracks[index].themeColourSlot = static_cast<int>(index);
        if (tracks[index].colorHex.trim().isEmpty())
            tracks[index].colorHex = defaultTrackColour(static_cast<int>(index)).toDisplayString(false);
    }

    return tracks;
}

void readAutomationLanes(const juce::var& lanesVar, std::vector<AutomationLane>& lanes)
{
    lanes.clear();
    if (auto* laneArray = lanesVar.getArray())
    {
        lanes.reserve(static_cast<size_t>(laneArray->size()));
        for (const auto& laneVar : *laneArray)
        {
            AutomationLane lane;
            lane.target = getStringProperty(laneVar, "target");
            lane.enabled = getBoolProperty(laneVar, "enabled", true);
            if (auto* pointsArray = getObjectProperty(laneVar, "points").getArray())
            {
                lane.points.reserve(static_cast<size_t>(pointsArray->size()));
                for (const auto& pointVar : *pointsArray)
                {
                    AutomationPoint point;
                    point.tick = getIntProperty(pointVar, "tick", 0, 0);
                    point.value = getDoubleProperty(pointVar, "value", 0.0);
                    lane.points.push_back(point);
                }
            }
            lanes.push_back(std::move(lane));
        }
    }
}

void readVstParameters(const juce::var& parametersVar, juce::NamedValueSet& parameters)
{
    parameters.clear();
    if (auto* object = parametersVar.getDynamicObject())
    {
        const auto& properties = object->getProperties();
        for (int index = 0; index < properties.size(); ++index)
        {
            const auto name = properties.getName(index).toString();
            const auto value = static_cast<double>(properties.getValueAt(index));
            if (std::isfinite(value))
                parameters.set(juce::Identifier(name), value);
        }
    }
}

void sanitiseMidiNoteList(std::vector<MidiNote>& notes)
{
    for (auto& note : notes)
    {
        note.startTick = juce::jmax(0, note.startTick);
        note.durationTick = juce::jmax(1, note.durationTick);
        note.pitch = juce::jlimit(0, 127, note.pitch);
        note.velocity = juce::jlimit(1, 127, note.velocity);
    }

    std::sort(notes.begin(),
              notes.end(),
              [] (const MidiNote& lhs, const MidiNote& rhs)
              {
                  if (lhs.startTick != rhs.startTick)
                      return lhs.startTick < rhs.startTick;
                  if (lhs.pitch != rhs.pitch)
                      return lhs.pitch > rhs.pitch;
                  return lhs.durationTick < rhs.durationTick;
              });
}

void sanitiseControllerLanePoints(AutomationLane& lane, bool clampToMidiRange)
{
    std::sort(lane.points.begin(),
              lane.points.end(),
              [] (const AutomationPoint& lhs, const AutomationPoint& rhs)
              {
                  return lhs.tick < rhs.tick;
              });

    std::vector<AutomationPoint> sanitised;
    sanitised.reserve(lane.points.size());
    for (const auto& point : lane.points)
    {
        AutomationPoint copy;
        copy.tick = juce::jmax(0, point.tick);
        copy.value = std::isfinite(point.value) ? point.value : 0.0;
        if (clampToMidiRange)
            copy.value = clampValue(copy.value, 0.0, 127.0);

        if (!sanitised.empty() && sanitised.back().tick == copy.tick)
            sanitised.back().value = copy.value;
        else
            sanitised.push_back(copy);
    }

    lane.points = std::move(sanitised);
}

void migrateLegacyTrackNotesToPatterns(ProjectState& project)
{
    if (!project.midiPatterns.empty())
        return;

    const auto projectBarTicks = ticksPerBar(project);

    for (int trackIndex = 0; trackIndex < static_cast<int>(project.tracks.size()); ++trackIndex)
    {
        auto& track = project.tracks[static_cast<size_t>(trackIndex)];
        if (track.notes.empty())
            continue;

        MidiPattern pattern;
        pattern.id = juce::Uuid().toString();
        pattern.name = track.name.trim().isNotEmpty() ? track.name + " Pattern" : "Pattern " + juce::String(trackIndex + 1);
        pattern.notes = track.notes;
        sanitiseMidiNoteList(pattern.notes);

        int lastTick = projectBarTicks;
        for (const auto& note : pattern.notes)
            lastTick = juce::jmax(lastTick, note.startTick + note.durationTick);
        pattern.lengthTicks = juce::jmax(projectBarTicks, lastTick);
        project.midiPatterns.push_back(pattern);

        bool attachedToExistingSection = false;
        for (auto& section : project.midiSections)
        {
            if (section.trackIndex != trackIndex || section.patternId.trim().isNotEmpty())
                continue;

            section.patternId = pattern.id;
            section.lengthTicks = juce::jmax(kMinSequenceSnapTicks, section.lengthTicks);
            section.name = section.name.trim().isNotEmpty() ? section.name : pattern.name;
            attachedToExistingSection = true;
        }

        if (!attachedToExistingSection)
        {
            MidiSection section;
            section.trackIndex = trackIndex;
            section.startTick = 0;
            section.lengthTicks = pattern.lengthTicks;
            section.patternId = pattern.id;
            section.name = pattern.name;
            project.midiSections.push_back(section);
        }
    }
}

void sanitiseTempoMarkerList(std::vector<TempoMarker>& tempoMarkers, int fallbackBpm)
{
    const auto clampedFallbackBpm = clampValue(fallbackBpm, 20, 300);

    for (auto& marker : tempoMarkers)
    {
        marker.tick = juce::jmax(0, marker.tick);
        marker.bpm = clampValue(marker.bpm, 20, 300);
    }

    std::stable_sort(tempoMarkers.begin(), tempoMarkers.end(), [](const TempoMarker& a, const TempoMarker& b)
    {
        return a.tick < b.tick;
    });

    std::vector<TempoMarker> collapsed;
    collapsed.reserve(tempoMarkers.size() + 1);
    for (const auto& marker : tempoMarkers)
    {
        if (!collapsed.empty() && collapsed.back().tick == marker.tick)
            collapsed.back().bpm = marker.bpm;
        else
            collapsed.push_back(marker);
    }

    if (collapsed.empty())
    {
        collapsed.push_back({ 0, clampedFallbackBpm });
    }
    else if (collapsed.front().tick != 0)
    {
        collapsed.insert(collapsed.begin(), TempoMarker{ 0, collapsed.front().bpm });
    }

    collapsed.front().tick = 0;
    collapsed.front().bpm = clampValue(collapsed.front().bpm, 20, 300);
    tempoMarkers = std::move(collapsed);
}

std::vector<TempoMarker> effectiveTempoMarkers(const ProjectState& project)
{
    auto markers = project.tempoMarkers;
    sanitiseTempoMarkerList(markers, project.bpm);
    return markers;
}

double tickSpanToSeconds(double tickSpan, double bpm)
{
    return (tickSpan * 60.0) / (juce::jmax(1.0, bpm) * static_cast<double>(kTicksPerBeat));
}

double tickToSecondsForTempoMarkers(const std::vector<TempoMarker>& tempoMarkers, int tick)
{
    if (tick <= 0 || tempoMarkers.empty())
        return 0.0;

    double seconds = 0.0;
    int previousTick = 0;
    double previousBpm = static_cast<double>(tempoMarkers.front().bpm);

    for (size_t index = 1; index < tempoMarkers.size(); ++index)
    {
        const auto markerTick = tempoMarkers[index].tick;
        if (tick <= markerTick)
            break;

        seconds += tickSpanToSeconds(static_cast<double>(markerTick - previousTick), previousBpm);
        previousTick = markerTick;
        previousBpm = static_cast<double>(tempoMarkers[index].bpm);
    }

    seconds += tickSpanToSeconds(static_cast<double>(tick - previousTick), previousBpm);
    return seconds;
}

double secondsToTickDoubleForTempoMarkers(const std::vector<TempoMarker>& tempoMarkers, double seconds)
{
    if (seconds <= 0.0 || tempoMarkers.empty())
        return 0.0;

    double remainingSeconds = seconds;
    int previousTick = 0;
    double previousBpm = static_cast<double>(tempoMarkers.front().bpm);

    for (size_t index = 1; index < tempoMarkers.size(); ++index)
    {
        const auto markerTick = tempoMarkers[index].tick;
        const auto segmentSeconds = tickSpanToSeconds(static_cast<double>(markerTick - previousTick), previousBpm);
        if (remainingSeconds < segmentSeconds)
            break;

        remainingSeconds -= segmentSeconds;
        previousTick = markerTick;
        previousBpm = static_cast<double>(tempoMarkers[index].bpm);
    }

    const auto ticksPerSecond = (previousBpm * static_cast<double>(kTicksPerBeat)) / 60.0;
    return static_cast<double>(previousTick) + (remainingSeconds * ticksPerSecond);
}

int64_t tickToFrameForTempoMarkers(const std::vector<TempoMarker>& tempoMarkers, int tick, double sampleRate)
{
    if (tick <= 0 || sampleRate <= 0.0 || tempoMarkers.empty())
        return 0;

    const auto seconds = tickToSecondsForTempoMarkers(tempoMarkers, tick);
    return static_cast<int64_t>(seconds * sampleRate);
}

double frameToTickDoubleForTempoMarkers(const std::vector<TempoMarker>& tempoMarkers, int64_t frame, double sampleRate)
{
    if (frame <= 0 || sampleRate <= 0.0 || tempoMarkers.empty())
        return 0.0;

    const auto seconds = static_cast<double>(frame) / sampleRate;
    return secondsToTickDoubleForTempoMarkers(tempoMarkers, seconds);
}
} // namespace

double tickToSeconds(int tick, int bpm)
{
    return (static_cast<double>(juce::jmax(0, tick)) * 60.0)
        / (static_cast<double>(juce::jmax(1, bpm)) * static_cast<double>(kTicksPerBeat));
}

int secondsToTick(double seconds, int bpm)
{
    const auto clampedSeconds = juce::jmax(0.0, seconds);
    const auto secondsPerTick = 60.0 / (static_cast<double>(juce::jmax(1, bpm)) * static_cast<double>(kTicksPerBeat));
    return static_cast<int>(clampedSeconds / secondsPerTick);
}

void sanitiseTempoMarkers(std::vector<TempoMarker>& tempoMarkers, int fallbackBpm)
{
    sanitiseTempoMarkerList(tempoMarkers, fallbackBpm);
}

double tickToSeconds(const ProjectState& project, int tick)
{
    return tickToSecondsForTempoMarkers(effectiveTempoMarkers(project), juce::jmax(0, tick));
}

int secondsToTick(const ProjectState& project, double seconds)
{
    return static_cast<int>(secondsToTickDoubleForTempoMarkers(effectiveTempoMarkers(project), juce::jmax(0.0, seconds)));
}

int64_t tickToFrame(const ProjectState& project, int tick, double sampleRate)
{
    return tickToFrameForTempoMarkers(effectiveTempoMarkers(project), juce::jmax(0, tick), sampleRate);
}

double frameToTickDouble(const ProjectState& project, int64_t frame, double sampleRate)
{
    return frameToTickDoubleForTempoMarkers(effectiveTempoMarkers(project), juce::jmax<int64_t>(0, frame), sampleRate);
}

int frameToTick(const ProjectState& project, int64_t frame, double sampleRate)
{
    return static_cast<int>(frameToTickDouble(project, frame, sampleRate));
}

const juce::StringArray& trackColorPalette()
{
    static const juce::StringArray palette {
        "#4AB4FF",
        "#FF8A65",
        "#7ED957",
        "#F4D35E",
        "#C084FC",
        "#FF6F91",
        "#4FD1C5",
        "#F97316"
    };

    return palette;
}

bool tryParseTrackHexColour(juce::String hex, juce::Colour& colour)
{
    hex = hex.trim();
    if (hex.startsWithChar('#'))
        hex = hex.substring(1);

    if (hex.length() == 6 && hex.containsOnly("0123456789abcdefABCDEF"))
    {
        const auto value = static_cast<juce::uint32>(hex.getHexValue32());
        colour = juce::Colour::fromRGB(static_cast<juce::uint8>((value >> 16) & 0xffu),
                                       static_cast<juce::uint8>((value >> 8) & 0xffu),
                                       static_cast<juce::uint8>(value & 0xffu));
        return true;
    }

    if (hex.length() == 8 && hex.containsOnly("0123456789abcdefABCDEF"))
    {
        const auto value = static_cast<juce::uint32>(hex.getHexValue32());
        colour = juce::Colour::fromRGBA(static_cast<juce::uint8>((value >> 16) & 0xffu),
                                        static_cast<juce::uint8>((value >> 8) & 0xffu),
                                        static_cast<juce::uint8>(value & 0xffu),
                                        static_cast<juce::uint8>((value >> 24) & 0xffu));
        return true;
    }

    return false;
}

bool isLegacyAutoTrackColour(const juce::String& value)
{
    juce::Colour candidate;
    if (!tryParseTrackHexColour(value, candidate))
        return false;

    const auto candidateHex = candidate.toDisplayString(false);
    for (const auto& paletteEntry : trackColorPalette())
    {
        juce::Colour paletteColour;
        if (tryParseTrackHexColour(paletteEntry, paletteColour)
            && paletteColour.toDisplayString(false).equalsIgnoreCase(candidateHex))
        {
            return true;
        }
    }

    return false;
}

juce::Colour applyThemeTrackTint(const juce::Colour& baseColour)
{
    const auto& theme = trackColourThemeState();
    auto colour = baseColour.interpolatedWith(theme.surface, 0.08f);
    colour = colour.interpolatedWith(theme.outline, 0.08f);

    const auto saturation = juce::jlimit(0.34f,
                                         0.9f,
                                         (colour.getSaturation() * 0.92f) + 0.04f);
    const auto brightness = juce::jlimit(0.42f,
                                         0.94f,
                                         (colour.getBrightness() * 0.9f) + 0.05f);
    return juce::Colour::fromHSV(colour.getHue(), saturation, brightness, 1.0f);
}

void setTrackColourTheme(const juce::Colour& accent,
                         const juce::Colour& surface,
                         const juce::Colour& outline,
                         const juce::Colour& success,
                         const juce::Colour& info,
                         const juce::Colour& warning)
{
    auto& theme = trackColourThemeState();
    theme.accent = accent;
    theme.surface = surface;
    theme.outline = outline;
    theme.success = success.isTransparent() ? rotateTrackHue(accent, 0.28f) : success;
    theme.info = info.isTransparent() ? rotateTrackHue(accent, 0.12f) : info;
    theme.warning = warning.isTransparent() ? rotateTrackHue(accent, -0.14f) : warning;
}

juce::Colour defaultTrackColour(int index)
{
    const auto palette = themedTrackPalette();
    const auto colourCount = juce::jmax(1, static_cast<int>(palette.size()));
    const auto colourIndex = ((index % colourCount) + colourCount) % colourCount;
    return applyThemeTrackTint(palette[static_cast<size_t>(colourIndex)]);
}

juce::Colour trackDisplayColour(const TrackState& track, int index)
{
    if (track.followThemeTrackColour)
        return defaultTrackColour(track.themeColourSlot >= 0 ? track.themeColourSlot : index);

    juce::Colour explicitColour;
    if (tryParseTrackHexColour(track.colorHex, explicitColour))
        return explicitColour;

    return defaultTrackColour(index);
}

juce::Colour trackTextColour(const juce::Colour& colour)
{
    const auto luminance = (0.299 * static_cast<double>(colour.getRed()))
        + (0.587 * static_cast<double>(colour.getGreen()))
        + (0.114 * static_cast<double>(colour.getBlue()));
    return luminance >= 170.0 ? juce::Colour::fromRGB(18, 18, 18)
                              : juce::Colour::fromRGB(242, 242, 242);
}

juce::String automationTargetForVstParameter(const juce::String& name)
{
    return juce::String(kAutomationTargetVstParameterPrefix) + name.trim();
}

bool automationTargetIsVstParameter(const juce::String& target)
{
    return target.trim().startsWithIgnoreCase(kAutomationTargetVstParameterPrefix);
}

juce::String automationTargetParameterName(const juce::String& target)
{
    const auto trimmed = target.trim();
    if (!trimmed.startsWithIgnoreCase(kAutomationTargetVstParameterPrefix))
        return {};

    return trimmed.substring(static_cast<int>(std::strlen(kAutomationTargetVstParameterPrefix))).trim();
}

int normaliseTimeSignatureNumerator(int numerator)
{
    return clampValue(numerator, 1, 32);
}

int normaliseTimeSignatureDenominator(int denominator)
{
    return nearestTimeSignatureDenominator(denominator);
}

int ticksPerTimeSignatureBeat(int denominator)
{
    const auto safeDenominator = normaliseTimeSignatureDenominator(denominator);
    return juce::jmax(1, (kTicksPerBeat * 4) / safeDenominator);
}

int ticksPerTimeSignatureBeat(const ProjectState& project)
{
    return ticksPerTimeSignatureBeat(project.timeSigDenominator);
}

int ticksPerBar(int numerator, int denominator)
{
    return juce::jmax(kMinSequenceSnapTicks,
                      normaliseTimeSignatureNumerator(numerator) * ticksPerTimeSignatureBeat(denominator));
}

int ticksPerBar(const ProjectState& project)
{
    return ticksPerBar(project.timeSigNumerator, project.timeSigDenominator);
}

juce::String timeSignatureDisplayName(int numerator, int denominator)
{
    return juce::String(normaliseTimeSignatureNumerator(numerator))
        + "/"
        + juce::String(normaliseTimeSignatureDenominator(denominator));
}

juce::String timeSignatureDisplayName(const ProjectState& project)
{
    return timeSignatureDisplayName(project.timeSigNumerator, project.timeSigDenominator);
}

int normaliseSequenceTickLength(int tickLength, int barTicks)
{
    const auto safeBarTicks = juce::jmax(kMinSequenceSnapTicks, barTicks);
    const std::array<int, 10> allowedTickLengths { {
        kTicksPerBeat / 16,
        kTicksPerBeat / 8,
        kTicksPerBeat / 4,
        kTicksPerBeat / 2,
        kTicksPerBeat,
        kTicksPerBeat * 2,
        safeBarTicks,
        safeBarTicks * 2,
        safeBarTicks * 4,
        safeBarTicks * 8
    } };

    auto best = safeBarTicks;
    auto bestDistance = std::numeric_limits<int>::max();
    for (const auto allowed : allowedTickLengths)
    {
        const auto distance = std::abs(tickLength - allowed);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            best = allowed;
        }
    }

    return juce::jmax(kMinSequenceSnapTicks, best);
}

int defaultPatternLengthTicks(const ProjectState& project)
{
    return normaliseSequenceTickLength(project.defaultPatternTicks, ticksPerBar(project));
}

int arrangementSnapTickLength(const ProjectState& project)
{
    return normaliseSequenceTickLength(project.arrangementSnapTicks, ticksPerBar(project));
}

juce::String editorToolModeLabel(EditorToolMode mode)
{
    switch (mode)
    {
        case EditorToolMode::pencil: return "Pencil";
        case EditorToolMode::selection: return "Select";
        case EditorToolMode::scissors: return "Scissors";
        case EditorToolMode::glue: return "Glue";
        case EditorToolMode::eraser: return "Eraser";
    }

    return "Pencil";
}

juce::String pianoRollInsertModeLabel(PianoRollInsertMode mode)
{
    switch (mode)
    {
        case PianoRollInsertMode::singleNote: return "Single Note";
        case PianoRollInsertMode::majorTriad: return "Major Triad";
        case PianoRollInsertMode::minorTriad: return "Minor Triad";
        case PianoRollInsertMode::majorFifth: return "Major 5th";
        case PianoRollInsertMode::minorFifth: return "Minor 5th";
        case PianoRollInsertMode::majorSeventh: return "Major 7th";
        case PianoRollInsertMode::minorSeventh: return "Minor 7th";
        case PianoRollInsertMode::dominantSeventh: return "Dominant 7th";
        case PianoRollInsertMode::suspended2: return "Suspended 2nd";
        case PianoRollInsertMode::suspended4: return "Suspended 4th";
        case PianoRollInsertMode::resolvedMajor: return "Resolved Major";
        case PianoRollInsertMode::resolvedMinor: return "Resolved Minor";
    }

    return "Single Note";
}

juce::StringArray availableKeyQuantizeScales()
{
    juce::StringArray scales;
    for (const auto& definition : keyQuantizeScaleDefinitions())
        scales.add(definition.id);
    return scales;
}

juce::String normaliseKeyQuantizeScale(const juce::String& scaleId)
{
    if (const auto* definition = findKeyQuantizeScaleDefinition(scaleId))
        return definition->id;
    return "chromatic";
}

juce::String pitchClassDisplayName(int pitchClass)
{
    static constexpr const char* names[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
    return names[juce::negativeAwareModulo(pitchClass, 12)];
}

juce::String keyQuantizeDisplayName(int rootPitchClass, const juce::String& scaleId)
{
    const auto normalisedScale = normaliseKeyQuantizeScale(scaleId);
    if (normalisedScale.equalsIgnoreCase("chromatic"))
        return "All Notes (Chromatic)";

    const auto* definition = findKeyQuantizeScaleDefinition(normalisedScale);
    if (definition == nullptr)
        return "All Notes (Chromatic)";

    return pitchClassDisplayName(rootPitchClass) + " " + definition->label;
}

juce::String midiControllerTarget(int controllerNumber)
{
    return "cc:" + juce::String(juce::jlimit(0, 127, controllerNumber));
}

bool controllerTargetIsMidiCc(const juce::String& target)
{
    return target.trim().startsWithIgnoreCase("cc:");
}

int midiControllerNumberFromTarget(const juce::String& target)
{
    if (!controllerTargetIsMidiCc(target))
        return -1;

    return juce::jlimit(0,
                        127,
                        target.fromFirstOccurrenceOf(":", false, false).trim().getIntValue());
}

juce::String midiControllerTargetLabel(const juce::String& target)
{
    const auto controllerNumber = midiControllerNumberFromTarget(target);
    if (controllerNumber < 0)
        return {};

    if (const auto* definition = findMidiControllerDefinition(controllerNumber))
        return "MIDI CC " + juce::String(controllerNumber) + ": " + definition->label;

    return "MIDI CC " + juce::String(controllerNumber);
}

juce::StringArray defaultMidiControllerTargets()
{
    juce::StringArray targets;
    for (const auto& definition : midiControllerDefinitions())
        targets.add(midiControllerTarget(definition.controller));
    return targets;
}

bool projectUsesKeyQuantize(const ProjectState& project)
{
    return !normaliseKeyQuantizeScale(project.keyQuantizeScale).equalsIgnoreCase("chromatic");
}

bool pitchInKeyQuantizeScale(const ProjectState& project, int pitch)
{
    const auto* definition = findKeyQuantizeScaleDefinition(project.keyQuantizeScale);
    if (definition == nullptr)
        return true;

    const auto root = juce::negativeAwareModulo(project.keyQuantizeRoot, 12);
    const auto pitchClass = juce::negativeAwareModulo(pitch, 12);
    for (const auto interval : definition->intervals)
    {
        if (juce::negativeAwareModulo(root + interval, 12) == pitchClass)
            return true;
    }

    return false;
}

std::vector<int> visiblePitchesForProjectScale(const ProjectState& project, int minPitch, int maxPitch)
{
    const auto lower = juce::jmin(minPitch, maxPitch);
    const auto upper = juce::jmax(minPitch, maxPitch);

    std::vector<int> pitches;
    pitches.reserve(static_cast<size_t>((upper - lower) + 1));

    for (int pitch = upper; pitch >= lower; --pitch)
    {
        if (pitchInKeyQuantizeScale(project, pitch))
            pitches.push_back(pitch);
    }

    if (pitches.empty())
    {
        for (int pitch = upper; pitch >= lower; --pitch)
            pitches.push_back(pitch);
    }

    return pitches;
}

double automationTargetDefaultValue(const TrackState& track, const juce::String& target)
{
    if (target.equalsIgnoreCase(kAutomationTargetVolume))
        return track.volume;
    if (target.equalsIgnoreCase(kAutomationTargetPan))
        return track.pan;
    if (target.equalsIgnoreCase(kAutomationTargetVstOutputGainDb))
        return track.vstiOutputGainDb;

    if (automationTargetIsVstParameter(target))
    {
        const auto parameterName = automationTargetParameterName(target);
        for (int index = 0; index < track.vstiParameters.size(); ++index)
        {
            if (track.vstiParameters.getName(index).toString().equalsIgnoreCase(parameterName))
                return static_cast<double>(track.vstiParameters.getValueAt(index));
        }
    }

    return 0.0;
}

std::pair<double, double> automationTargetBounds(const TrackState& track, const juce::String& target)
{
    juce::ignoreUnused(track);

    if (target.equalsIgnoreCase(kAutomationTargetVolume))
        return { 0.0, 1.5 };
    if (target.equalsIgnoreCase(kAutomationTargetPan))
        return { -1.0, 1.0 };
    if (target.equalsIgnoreCase(kAutomationTargetVstOutputGainDb))
        return { -48.0, 24.0 };
    if (automationTargetIsVstParameter(target))
        return { 0.0, 100.0 };
    return { 0.0, 1.0 };
}

juce::String automationTargetLabel(const TrackState& track, const juce::String& target)
{
    if (target.equalsIgnoreCase(kAutomationTargetVolume))
        return "Track Volume";
    if (target.equalsIgnoreCase(kAutomationTargetPan))
        return "Track Pan";
    if (target.equalsIgnoreCase(kAutomationTargetVstOutputGainDb))
    {
        return track.instrumentMode.trim().equalsIgnoreCase("VSTI Rack")
            ? "VST Output Gain"
            : "Output Gain";
    }
    if (automationTargetIsVstParameter(target))
    {
        const auto parameterName = automationTargetParameterName(target);
        return "VST Parameter: " + (parameterName.isNotEmpty() ? parameterName : "Unnamed");
    }

    return target.isNotEmpty() ? target : "Automation";
}

juce::StringArray availableAutomationTargets(const TrackState& track)
{
    juce::StringArray targets;
    targets.add(kAutomationTargetVolume);
    targets.add(kAutomationTargetPan);

    if (track.trackType.trim().equalsIgnoreCase("instrument")
        && track.instrumentMode.trim().equalsIgnoreCase("VSTI Rack"))
    {
        targets.add(kAutomationTargetVstOutputGainDb);

        juce::StringArray seenParameters;
        for (int index = 0; index < track.vstiParameters.size(); ++index)
        {
            const auto parameterName = track.vstiParameters.getName(index).toString().trim();
            if (parameterName.isEmpty() || seenParameters.contains(parameterName, true))
                continue;

            seenParameters.add(parameterName);
            targets.add(automationTargetForVstParameter(parameterName));
        }
    }

    return targets;
}

juce::StringArray usedAutomationTargets(const TrackState& track)
{
    juce::StringArray targets;
    for (const auto& lane : track.automationLanes)
    {
        const auto target = lane.target.trim();
        if (target.isEmpty() || lane.points.empty())
            continue;

        targets.addIfNotAlreadyThere(target);
    }

    return targets;
}

juce::StringArray visibleArrangementAutomationTargets(const TrackState& track)
{
    juce::StringArray visibleTargets;
    const auto usedTargets = usedAutomationTargets(track);

    for (const auto& entry : track.arrangementVisibleAutomationTargets)
    {
        const auto target = entry.trim();
        if (target.isEmpty() || !usedTargets.contains(target, true))
            continue;

        visibleTargets.addIfNotAlreadyThere(target);
    }

    return visibleTargets;
}

void sanitiseAutomationLanes(TrackState& track)
{
    std::vector<AutomationLane> sanitised;
    juce::StringArray seenTargets;
    const auto availableTargets = availableAutomationTargets(track);

    for (const auto& lane : track.automationLanes)
    {
        const auto target = lane.target.trim();
        if (target.isEmpty())
            continue;
        if (!automationTargetIsVstParameter(target) && !availableTargets.contains(target, true))
            continue;
        if (seenTargets.contains(target, true))
            continue;

        seenTargets.add(target);
        const auto [minimum, maximum] = automationTargetBounds(track, target);
        const auto fallbackValue = automationTargetDefaultValue(track, target);

        std::vector<AutomationPoint> points = lane.points;
        std::sort(points.begin(), points.end(), [] (const AutomationPoint& lhs, const AutomationPoint& rhs)
        {
            if (lhs.tick != rhs.tick)
                return lhs.tick < rhs.tick;
            return lhs.value < rhs.value;
        });

        std::vector<AutomationPoint> dedupedPoints;
        dedupedPoints.reserve(points.size());
        for (const auto& point : points)
        {
            AutomationPoint normalisedPoint;
            normalisedPoint.tick = juce::jmax(0, point.tick);
            const auto value = std::isfinite(point.value) ? point.value : fallbackValue;
            normalisedPoint.value = juce::jlimit(minimum, maximum, value);

            if (!dedupedPoints.empty() && dedupedPoints.back().tick == normalisedPoint.tick)
                dedupedPoints.back().value = normalisedPoint.value;
            else
                dedupedPoints.push_back(normalisedPoint);
        }

        AutomationLane sanitisedLane;
        sanitisedLane.target = target;
        sanitisedLane.enabled = lane.enabled;
        sanitisedLane.points = std::move(dedupedPoints);
        sanitised.push_back(std::move(sanitisedLane));
    }

    track.automationLanes = std::move(sanitised);
}

void sanitiseArrangementAutomationLaneVisibility(TrackState& track)
{
    track.arrangementVisibleAutomationTargets = visibleArrangementAutomationTargets(track);
}

void sanitisePatternControllerLanes(MidiPattern& pattern)
{
    std::vector<AutomationLane> sanitised;
    juce::StringArray seenTargets;

    for (const auto& lane : pattern.controllerLanes)
    {
        const auto target = lane.target.trim();
        const auto controllerNumber = midiControllerNumberFromTarget(target);
        if (controllerNumber < 0)
            continue;
        if (seenTargets.contains(target, true))
            continue;

        AutomationLane copy = lane;
        copy.target = midiControllerTarget(controllerNumber);
        copy.enabled = lane.enabled;
        sanitiseControllerLanePoints(copy, true);
        seenTargets.add(copy.target);
        sanitised.push_back(std::move(copy));
    }

    pattern.controllerLanes = std::move(sanitised);
}

int defaultMidiProgramForInstrumentName(const juce::String& instrumentName)
{
    const auto normalized = instrumentName.trim().toLowerCase();
    const std::pair<const char*, int> defaults[] =
    {
        { "piano", 0 },
        { "chromatic", 8 },
        { "organ", 16 },
        { "guitar", 24 },
        { "bass", 32 },
        { "strings", 40 },
        { "ensemble", 48 },
        { "brass", 56 },
        { "reed", 64 },
        { "pipe", 72 },
        { "lead", 80 },
        { "pad", 88 },
        { "fx", 96 },
        { "ethnic", 104 },
        { "percussive", 112 },
        { "sfx", 120 },
    };

    for (const auto& [token, program] : defaults)
    {
        if (normalized.contains(token))
            return program;
    }

    return 0;
}

juce::String inferSynthProfile(const juce::String& instrumentName, int midiProgram)
{
    const auto normalized = instrumentName.trim().toLowerCase();
    const std::pair<const char*, const char*> profileMap[] =
    {
        { "piano", "e_piano" },
        { "chromatic", "e_piano" },
        { "organ", "organ" },
        { "guitar", "pluck" },
        { "bass", "sub_bass" },
        { "strings", "saw_pad" },
        { "brass", "brass_stack" },
        { "reed", "reed_breath" },
        { "pipe", "reed_breath" },
        { "lead", "synth" },
        { "pad", "saw_pad" },
        { "percussive", "noise_kit" },
        { "ensemble", "saw_pad" },
        { "fx", "synth" },
        { "ethnic", "pluck" },
        { "sfx", "synth" },
    };

    for (const auto& [token, profile] : profileMap)
    {
        if (normalized.contains(token))
            return profile;
    }

    const auto program = clampValue(midiProgram, 0, 127);
    if (program < 16)
        return "e_piano";
    if (program < 24)
        return "organ";
    if (program < 32)
        return "pluck";
    if (program < 40)
        return "sub_bass";
    if (program < 56)
        return "saw_pad";
    if (program < 64)
        return "brass_stack";
    if (program < 80)
        return "reed_breath";
    if (program < 120)
        return "synth";
    return "noise_kit";
}

int findRackInstrumentIndexByReference(const ProjectState& project, const juce::String& reference)
{
    const auto trimmedReference = reference.trim();
    if (trimmedReference.isEmpty())
        return -1;

    const auto referenceFile = juce::File(trimmedReference);
    const auto referencePath = referenceFile.getFullPathName();
    const auto referenceFileName = referenceFile.getFileName();
    const auto referenceStem = referenceFile.getFileNameWithoutExtension();

    for (int index = 0; index < static_cast<int>(project.vstRack.size()); ++index)
    {
        const auto& entry = project.vstRack[static_cast<size_t>(index)];
        const auto entryPath = entry.path.trim();
        const auto entryName = entry.name.trim();
        const auto entryPluginName = entry.pluginName.trim();

        if (trimmedReference.equalsIgnoreCase(entryPath)
            || trimmedReference.equalsIgnoreCase(entryName)
            || trimmedReference.equalsIgnoreCase(entryPluginName))
        {
            return index;
        }

        if (entryPath.isNotEmpty())
        {
            const auto entryFile = juce::File(entryPath);
            if (referencePath.equalsIgnoreCase(entryFile.getFullPathName())
                || trimmedReference.equalsIgnoreCase(entryFile.getFileName())
                || trimmedReference.equalsIgnoreCase(entryFile.getFileNameWithoutExtension())
                || referenceFileName.equalsIgnoreCase(entryFile.getFileName())
                || referenceStem.equalsIgnoreCase(entryFile.getFileNameWithoutExtension()))
            {
                return index;
            }
        }
    }

    return -1;
}

juce::String resolveRackPluginPath(const ProjectState& project, const TrackState& track)
{
    const auto directReference = track.rackVst.trim();
    if (directReference.isNotEmpty() && juce::File(directReference).exists())
        return juce::File(directReference).getFullPathName();

    if (const auto rackIndex = findRackInstrumentIndexByReference(project, directReference); rackIndex >= 0)
    {
        const auto entryPath = project.vstRack[static_cast<size_t>(rackIndex)].path.trim();
        if (entryPath.isNotEmpty() && juce::File(entryPath).exists())
            return juce::File(entryPath).getFullPathName();
    }

    if (const auto instrumentIndex = findRackInstrumentIndexByReference(project, track.instrument); instrumentIndex >= 0)
    {
        const auto entryPath = project.vstRack[static_cast<size_t>(instrumentIndex)].path.trim();
        if (entryPath.isNotEmpty() && juce::File(entryPath).exists())
            return juce::File(entryPath).getFullPathName();
    }

    return {};
}

juce::String displayRackName(const ProjectState& project, const TrackState& track)
{
    if (const auto rackIndex = findRackInstrumentIndexByReference(project, track.rackVst); rackIndex >= 0)
    {
        const auto& entry = project.vstRack[static_cast<size_t>(rackIndex)];
        if (entry.name.isNotEmpty())
            return entry.name;
        if (entry.pluginName.isNotEmpty())
            return entry.pluginName;
        if (entry.path.isNotEmpty())
            return juce::File(entry.path).getFileNameWithoutExtension();
    }

    const auto directReference = track.rackVst.trim();
    if (directReference.isNotEmpty())
    {
        const auto file = juce::File(directReference);
        if (file.getFileName().isNotEmpty() && directReference.containsAnyOf("\\/"))
            return file.getFileNameWithoutExtension();
        return directReference;
    }

    return track.instrument.trim();
}

int preferredInstrumentIndexByNames(const ProjectState& project, const juce::StringArray& preferredNames)
{
    for (const auto& preferredName : preferredNames)
    {
        const auto trimmedName = preferredName.trim();
        if (trimmedName.isEmpty())
            continue;

        for (int index = 0; index < static_cast<int>(project.vstRack.size()); ++index)
        {
            const auto& entry = project.vstRack[static_cast<size_t>(index)];
            if (!entry.hostSupported || !entry.isInstrument)
                continue;

            if (trimmedName.equalsIgnoreCase(entry.name)
                || trimmedName.equalsIgnoreCase(entry.pluginName)
                || trimmedName.equalsIgnoreCase(juce::File(entry.path).getFileNameWithoutExtension()))
            {
                return index;
            }
        }
    }

    return -1;
}

int defaultInstrumentIndex(const ProjectState& project)
{
    const juce::StringArray preferredNames
    {
        "AI Piano",
        "AI Strings",
        "AI Bass Guitar",
        "AI Organ",
        "AI Flute",
        "AI Saxophone",
        "AI Violin",
        "AI Bass Synth",
        "AI Lead Synth",
        "AI Pad Synth",
        "AI Pluck Synth",
        "AI String Synth",
        "Surge XT",
    };

    if (const auto preferredIndex = preferredInstrumentIndexByNames(project, preferredNames); preferredIndex >= 0)
        return preferredIndex;

    for (int index = 0; index < static_cast<int>(project.vstRack.size()); ++index)
    {
        const auto& entry = project.vstRack[static_cast<size_t>(index)];
        if (entry.hostSupported && entry.isInstrument)
            return index;
    }

    return -1;
}

int suggestedNativeInstrumentIndexForTrack(const ProjectState& project, const TrackState& track)
{
    if (!track.trackType.trim().equalsIgnoreCase("instrument"))
        return -1;

    if (track.instrumentMode.trim().equalsIgnoreCase("VSTI Rack"))
    {
        const auto existingIndex = findRackInstrumentIndexByReference(project, track.rackVst);
        if (existingIndex >= 0)
        {
            const auto& entry = project.vstRack[static_cast<size_t>(existingIndex)];
            if (entry.hostSupported && entry.isInstrument)
                return existingIndex;
        }
    }

    const auto instrumentName = track.instrument.trim().toLowerCase();
    const auto family = track.synthProfile.trim().toLowerCase();
    const auto midiProgram = clampValue(track.midiProgram, 0, 127);
    const auto midiChannel = clampValue(track.midiChannel, 0, 15);

    juce::StringArray preferredNames;
    if (midiChannel == 9 || family == "drums" || family == "noise_kit"
        || instrumentName.contains("drum") || instrumentName.contains("kit"))
    {
        preferredNames.add("AI Drum Machine");
    }
    else if (family == "bass" || family == "sub_bass"
             || (midiProgram >= 32 && midiProgram <= 39) || instrumentName.contains("bass"))
    {
        preferredNames.addArray(juce::StringArray{ "AI Bass Guitar", "AI Bass Synth", "AI Lead Synth" });
    }
    else if (family == "strings" || family == "saw_pad"
             || instrumentName.contains("string") || instrumentName.contains("violin")
             || instrumentName.contains("cello") || instrumentName.contains("pad"))
    {
        preferredNames.addArray(juce::StringArray{ "AI Strings", "AI Violin", "AI String Synth", "AI Pad Synth" });
    }
    else if (family == "guitar" || family == "pluck"
             || instrumentName.contains("guitar") || instrumentName.contains("pluck")
             || instrumentName.contains("harp") || instrumentName.contains("mallet"))
    {
        preferredNames.addArray(juce::StringArray{ "AI Guitar", "AI Harp", "AI Pluck Synth", "AI Lead Synth" });
    }
    else if (family == "piano" || family == "organ" || family == "e_piano"
             || instrumentName.contains("piano") || instrumentName.contains("keys")
             || instrumentName.contains("organ") || instrumentName.contains("ep"))
    {
        preferredNames.addArray(juce::StringArray{ "AI Piano", "AI Organ", "AI Pluck Synth", "AI Pad Synth", "AI Lead Synth" });
    }
    else if (family == "horn" || family == "brass" || family == "woodwind"
             || family == "brass_stack" || family == "reed_breath"
             || instrumentName.contains("horn") || instrumentName.contains("brass")
             || instrumentName.contains("trumpet") || instrumentName.contains("trombone")
             || instrumentName.contains("sax") || instrumentName.contains("flute")
             || instrumentName.contains("lead"))
    {
        preferredNames.addArray(juce::StringArray{ "AI Saxophone", "AI Flute", "AI Lead Synth", "AI String Synth" });
    }
    else
    {
        preferredNames.addArray(juce::StringArray{ "AI Piano", "AI Strings", "AI Lead Synth", "AI Pad Synth", "AI String Synth", "AI Bass Synth" });
    }

    return preferredInstrumentIndexByNames(project, preferredNames);
}

int nativeInstrumentIndexForTrack(const ProjectState& project, const TrackState& track)
{
    if (const auto suggestedIndex = suggestedNativeInstrumentIndexForTrack(project, track); suggestedIndex >= 0)
        return suggestedIndex;

    if (!track.trackType.trim().equalsIgnoreCase("instrument"))
        return -1;

    return defaultInstrumentIndex(project);
}

bool materialiseNativeInstrumentTrack(const ProjectState& project, TrackState& track)
{
    if (!track.trackType.trim().equalsIgnoreCase("instrument"))
        return false;

    const auto instrumentIndex = nativeInstrumentIndexForTrack(project, track);
    if (instrumentIndex < 0)
        return false;

    const auto& entry = project.vstRack[static_cast<size_t>(instrumentIndex)];
    const auto instrumentLabel = track.instrument.trim();
    const auto entryLabel = entry.name.isNotEmpty() ? entry.name
                                                    : (entry.pluginName.isNotEmpty() ? entry.pluginName
                                                                                     : juce::File(entry.path).getFileNameWithoutExtension());

    track.instrumentMode = "VSTI Rack";
    track.rackVst = entryLabel;
    track.instrument = instrumentLabel.isNotEmpty() ? instrumentLabel : entryLabel;
    track.vstiStatePath.clear();
    track.vstiStateBase64.clear();
    track.vstiParameters.clear();
    track.synthProfile = "vst_instrument";
    return true;
}

int patternLengthTicks(const MidiPattern& pattern)
{
    auto lengthTicks = juce::jmax(kMinSequenceSnapTicks, pattern.lengthTicks);
    for (const auto& note : pattern.notes)
        lengthTicks = juce::jmax(lengthTicks, note.startTick + note.durationTick);
    for (const auto& lane : pattern.controllerLanes)
    {
        for (const auto& point : lane.points)
            lengthTicks = juce::jmax(lengthTicks, point.tick);
    }
    return lengthTicks;
}

MidiPattern* findMidiPattern(ProjectState& project, const juce::String& patternId)
{
    const auto trimmed = patternId.trim();
    if (trimmed.isEmpty())
        return nullptr;

    for (auto& pattern : project.midiPatterns)
    {
        if (pattern.id == trimmed)
            return &pattern;
    }

    return nullptr;
}

const MidiPattern* findMidiPattern(const ProjectState& project, const juce::String& patternId)
{
    const auto trimmed = patternId.trim();
    if (trimmed.isEmpty())
        return nullptr;

    for (const auto& pattern : project.midiPatterns)
    {
        if (pattern.id == trimmed)
            return &pattern;
    }

    return nullptr;
}

std::vector<MidiControllerEvent> collectTrackControllerEvents(const ProjectState& project, int trackIndex)
{
    std::vector<MidiControllerEvent> events;
    if (!juce::isPositiveAndBelow(trackIndex, static_cast<int>(project.tracks.size())))
        return events;

    const auto channel = juce::jlimit(1, 16, project.tracks[static_cast<size_t>(trackIndex)].midiChannel + 1);
    for (const auto& section : project.midiSections)
    {
        if (section.trackIndex != trackIndex)
            continue;

        const auto* pattern = findMidiPattern(project, section.patternId);
        if (pattern == nullptr)
            continue;

        for (const auto& lane : pattern->controllerLanes)
        {
            const auto controllerNumber = midiControllerNumberFromTarget(lane.target);
            if (!lane.enabled || controllerNumber < 0)
                continue;

            for (const auto& point : lane.points)
            {
                MidiControllerEvent event;
                event.tick = juce::jmax(0, section.startTick + point.tick);
                event.controller = controllerNumber;
                event.value = juce::jlimit(0, 127, juce::roundToInt(point.value));
                event.channel = channel;
                events.push_back(event);
            }
        }
    }

    std::sort(events.begin(),
              events.end(),
              [] (const MidiControllerEvent& lhs, const MidiControllerEvent& rhs)
              {
                  if (lhs.tick != rhs.tick)
                      return lhs.tick < rhs.tick;
                  if (lhs.controller != rhs.controller)
                      return lhs.controller < rhs.controller;
                  return lhs.channel < rhs.channel;
              });

    return events;
}

void synchronisePatternClipsToTrackNotes(ProjectState& project)
{
    for (auto& track : project.tracks)
        track.notes.clear();

    for (auto& pattern : project.midiPatterns)
    {
        if (pattern.id.trim().isEmpty())
            pattern.id = juce::Uuid().toString();
        pattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks, pattern.lengthTicks);
        sanitiseMidiNoteList(pattern.notes);
        sanitisePatternControllerLanes(pattern);
    }

    for (auto& section : project.midiSections)
    {
        section.trackIndex = juce::jmax(0, section.trackIndex);
        section.startTick = juce::jmax(0, section.startTick);
        section.lengthTicks = juce::jmax(kMinSequenceSnapTicks, section.lengthTicks);

        if (section.patternId.trim().isEmpty())
            continue;

        const auto* pattern = findMidiPattern(project, section.patternId);
        if (pattern == nullptr)
            continue;

        section.lengthTicks = patternLengthTicks(*pattern);
        section.name = section.name.trim().isNotEmpty() ? section.name : pattern->name;
        if (!juce::isPositiveAndBelow(section.trackIndex, static_cast<int>(project.tracks.size())))
            continue;

        auto& track = project.tracks[static_cast<size_t>(section.trackIndex)];
        track.notes.reserve(track.notes.size() + pattern->notes.size());
        for (const auto& patternNote : pattern->notes)
        {
            auto note = patternNote;
            note.startTick = juce::jmax(0, section.startTick + patternNote.startTick);
            note.selected = false;
            track.notes.push_back(std::move(note));
        }
    }

    for (auto& track : project.tracks)
        sanitiseMidiNoteList(track.notes);
}

static void removeUnusedPatterns(ProjectState& project)
{
    juce::StringArray referencedPatternIds;
    for (const auto& section : project.midiSections)
    {
        const auto patternId = section.patternId.trim();
        if (patternId.isNotEmpty())
            referencedPatternIds.addIfNotAlreadyThere(patternId);
    }

    project.midiPatterns.erase(std::remove_if(project.midiPatterns.begin(),
                                              project.midiPatterns.end(),
                                              [&referencedPatternIds] (const MidiPattern& pattern)
                                              {
                                                  return !referencedPatternIds.contains(pattern.id);
                                              }),
                               project.midiPatterns.end());
}

void ProjectState::recalculateTimeFields()
{
    bpm = clampValue(bpm, 20, 300);
    timeSigNumerator = normaliseTimeSignatureNumerator(timeSigNumerator);
    timeSigDenominator = normaliseTimeSignatureDenominator(timeSigDenominator);
    defaultPatternTicks = normaliseSequenceTickLength(defaultPatternTicks, ticksPerBar(*this));
    quantizeDiv = clampValue(quantizeDiv, 1, 64);
    keyQuantizeRoot = juce::negativeAwareModulo(keyQuantizeRoot, 12);
    keyQuantizeScale = normaliseKeyQuantizeScale(keyQuantizeScale);
    pianoRollVisiblePitchMin = clampValue(pianoRollVisiblePitchMin, kEditableMidiPitchMin, kEditableMidiPitchMax);
    pianoRollVisiblePitchMax = clampValue(pianoRollVisiblePitchMax, kEditableMidiPitchMin, kEditableMidiPitchMax);
    if (pianoRollVisiblePitchMin > pianoRollVisiblePitchMax)
        std::swap(pianoRollVisiblePitchMin, pianoRollVisiblePitchMax);
    arrangementSnapTicks = normaliseSequenceTickLength(arrangementSnapTicks, ticksPerBar(*this));
    leftLocatorTick = juce::jmax(0, leftLocatorTick);
    rightLocatorTick = juce::jmax(leftLocatorTick + 1, rightLocatorTick);
    playheadTick = clampValue(playheadTick, leftLocatorTick, rightLocatorTick);

    juce::StringArray validRoutingTargets;
    validRoutingTargets.add("master");
    validRoutingTargets.add("none");
    for (auto& bus : sharedFxBuses)
    {
        if (bus.id.trim().isEmpty())
            bus.id = juce::Uuid().toString();
        if (bus.name.trim().isEmpty())
            bus.name = "FX Bus";
        validRoutingTargets.addIfNotAlreadyThere(bus.id);
    }
    for (auto& bus : sharedFxBuses)
    {
        juce::StringArray sanitisedTargets;
        for (const auto& target : bus.outputTargets)
        {
            const auto trimmedTarget = target.trim();
            if (trimmedTarget.isEmpty() || trimmedTarget.equalsIgnoreCase(bus.id))
                continue;
            if (validRoutingTargets.contains(trimmedTarget, true))
                sanitisedTargets.addIfNotAlreadyThere(trimmedTarget);
        }
        bus.outputTargets = sanitisedTargets;
    }
    for (auto& track : tracks)
    {
        const auto target = track.routingTarget.trim();
        track.routingTarget = target.isEmpty()
            ? juce::String("none")
            : (validRoutingTargets.contains(target, true) ? target : "master");
    }

    for (auto& pattern : midiPatterns)
    {
        if (pattern.id.trim().isEmpty())
            pattern.id = juce::Uuid().toString();
        pattern.lengthTicks = juce::jmax(kMinSequenceSnapTicks, pattern.lengthTicks);
        sanitiseMidiNoteList(pattern.notes);
        sanitisePatternControllerLanes(pattern);
        int lastTick = kMinSequenceSnapTicks;
        for (const auto& note : pattern.notes)
            lastTick = juce::jmax(lastTick, note.startTick + note.durationTick);
        for (const auto& lane : pattern.controllerLanes)
        {
            for (const auto& point : lane.points)
                lastTick = juce::jmax(lastTick, point.tick);
        }
        pattern.lengthTicks = juce::jmax(pattern.lengthTicks, lastTick);
    }

    for (auto& section : midiSections)
    {
        section.trackIndex = juce::jmax(0, section.trackIndex);
        section.startTick = juce::jmax(0, section.startTick);
        if (const auto* pattern = findMidiPattern(*this, section.patternId))
        {
            section.lengthTicks = patternLengthTicks(*pattern);
            if (section.name.trim().isEmpty())
                section.name = pattern->name;
        }
        else
        {
            section.lengthTicks = juce::jmax(kMinSequenceSnapTicks, section.lengthTicks);
        }

        section.startSec = tickToSeconds(*this, section.startTick);
        section.durationSec = tickToSeconds(*this, juce::jmax(kMinSequenceSnapTicks, section.lengthTicks));
    }

    removeUnusedPatterns(*this);
    synchronisePatternClipsToTrackNotes(*this);
    leftLocatorSec = tickToSeconds(leftLocatorTick, bpm);
    rightLocatorSec = tickToSeconds(rightLocatorTick, bpm);
    playheadSec = tickToSeconds(playheadTick, bpm);
}

int ProjectState::getTotalNoteCount() const
{
    int noteCount = 0;
    for (const auto& track : tracks)
        noteCount += static_cast<int>(track.notes.size());
    return noteCount;
}

ProjectFileData makeDefaultProjectFile()
{
    ProjectFileData data;
    data.project.tracks = makeDefaultTracks();
    data.project.recalculateTimeFields();
    data.savedAtUnix = juce::Time::getCurrentTime().toMilliseconds() / 1000;
    data.uiState = makeObjectVar();
    return data;
}

juce::var toVar(const TrackState& track)
{
    return serialiseTrack(track);
}

juce::var toVar(const ProjectState& project)
{
    return serialiseProjectState(project);
}

juce::String fingerprint(const TrackState& track)
{
    return juce::JSON::toString(toVar(track), true);
}

juce::String fingerprint(const ProjectState& project)
{
    return juce::JSON::toString(toVar(project), true);
}

juce::Result loadProjectFile(const juce::File& file, ProjectFileData& outProject)
{
    if (!file.existsAsFile())
        return juce::Result::fail("Project file not found.");

    juce::var parsed;
    const auto parseResult = juce::JSON::parse(file.loadFileAsString(), parsed);
    if (parseResult.failed())
        return juce::Result::fail("Failed to parse JSON: " + parseResult.getErrorMessage());

    ProjectFileData loaded = makeDefaultProjectFile();
    loaded.format = getStringProperty(parsed, "format", "ai_music_studio_project");
    loaded.version = getIntProperty(parsed, "version", kProjectFileVersion, 1, 999);
    loaded.savedAtUnix = static_cast<juce::int64>(getDoubleProperty(parsed, "saved_at_unix", 0.0, 0.0));
    loaded.uiState = getObjectProperty(parsed, "ui");

    const auto projectRoot = getObjectProperty(parsed, "project", parsed);
    loaded.project.bpm = getIntProperty(projectRoot, "bpm", kDefaultBpm, 20, 300);
    loaded.project.timeSigNumerator = getIntProperty(projectRoot, "time_signature_numerator", 4, 1, 32);
    loaded.project.timeSigDenominator = normaliseTimeSignatureDenominator(getIntProperty(projectRoot,
                                                                                          "time_signature_denominator",
                                                                                          4,
                                                                                          1,
                                                                                          32));
    loaded.project.defaultPatternTicks = getIntProperty(projectRoot,
                                                        "default_pattern_ticks",
                                                        getIntProperty(projectRoot, "default_pattern_bars", 1, 1, 128) * ticksPerBar(loaded.project),
                                                        kMinSequenceSnapTicks);
    loaded.project.quantizeEnabled = getBoolProperty(projectRoot, "quantize_enabled", true);
    loaded.project.quantizeDiv = getIntProperty(projectRoot, "quantize_div", 8, 1, 64);
    loaded.project.quantizeTriplet = getBoolProperty(projectRoot, "quantize_triplet", false);
    loaded.project.keyQuantizeRoot = getIntProperty(projectRoot, "key_quantize_root", 0, 0, 11);
    loaded.project.keyQuantizeScale = getStringProperty(projectRoot, "key_quantize_scale", "chromatic");
    loaded.project.pianoRollVisiblePitchMin = getIntProperty(projectRoot,
                                                             "piano_roll_visible_pitch_min",
                                                             kEditableMidiPitchMin,
                                                             kEditableMidiPitchMin,
                                                             kEditableMidiPitchMax);
    loaded.project.pianoRollVisiblePitchMax = getIntProperty(projectRoot,
                                                             "piano_roll_visible_pitch_max",
                                                             kEditableMidiPitchMax,
                                                             kEditableMidiPitchMin,
                                                             kEditableMidiPitchMax);
    loaded.project.arrangementSnapTicks = getIntProperty(projectRoot, "arrangement_snap_ticks", kTicksPerBeat, kMinSequenceSnapTicks);
    loaded.project.loopEnabled = getBoolProperty(projectRoot, "loop_enabled", true);
    loaded.project.metronomeEnabled = getBoolProperty(projectRoot, "metronome_enabled", false);

    const auto defaultLeftSec = 0.0;
    const auto defaultRightSec = tickToSeconds(ticksPerBar(loaded.project), loaded.project.bpm);
    const auto leftSec = getDoubleProperty(projectRoot, "left_locator_sec", defaultLeftSec, 0.0, 3600.0);
    const auto rightSec = getDoubleProperty(projectRoot, "right_locator_sec", defaultRightSec, 0.0, 3600.0);
    const auto playheadSec = getDoubleProperty(projectRoot, "playhead_sec", 0.0, 0.0, juce::jmax(rightSec, 0.0));

    loaded.project.leftLocatorTick = getIntProperty(projectRoot, "left_locator_tick", secondsToTick(leftSec, loaded.project.bpm), 0);
    loaded.project.rightLocatorTick = getIntProperty(projectRoot, "right_locator_tick", secondsToTick(rightSec, loaded.project.bpm), 0);
    loaded.project.playheadTick = getIntProperty(projectRoot, "playhead_tick", secondsToTick(playheadSec, loaded.project.bpm), 0);

    loaded.project.tracks.clear();
    if (auto* trackArray = getObjectProperty(projectRoot, "tracks").getArray())
    {
        loaded.project.tracks.reserve(static_cast<size_t>(trackArray->size()));
        for (int trackIndex = 0; trackIndex < trackArray->size(); ++trackIndex)
        {
            const auto& trackVar = trackArray->getReference(trackIndex);
            TrackState track;
            track.name = getStringProperty(trackVar, "name", "Track " + juce::String(trackIndex + 1));
            track.trackType = getStringProperty(trackVar, "track_type", "instrument").trim().equalsIgnoreCase("sample")
                ? "sample"
                : "instrument";
            track.volume = getDoubleProperty(trackVar, "volume", 0.8, 0.0, 2.0);
            track.pan = getDoubleProperty(trackVar, "pan", 0.0, -1.0, 1.0);
            track.instrument = getStringProperty(trackVar, "instrument", track.instrument);
            track.instrumentMode = getStringProperty(trackVar, "instrument_mode", track.instrumentMode);
            track.rackVst = getStringProperty(trackVar, "rack_vsti");
            track.plugins = readStringArray(getObjectProperty(trackVar, "plugins"));
            readVstParameters(getObjectProperty(trackVar, "vsti_parameters"), track.vstiParameters);
            track.vstiStatePath = getStringProperty(trackVar, "vsti_state_path");
            track.vstiStateBase64 = getStringProperty(trackVar, "vsti_state_b64");
            track.vstiOutputGainDb = getDoubleProperty(trackVar, "vsti_output_gain_db", 0.0, -48.0, 24.0);
            track.vstiWetMix = getDoubleProperty(trackVar, "vsti_wet_mix", 100.0, 0.0, 100.0);
            track.vstiOutputBusRoutes = readVarArray(getObjectProperty(trackVar, "vsti_output_bus_routes"));
            track.vstiInputBusRoutes = readVarArray(getObjectProperty(trackVar, "vsti_input_bus_routes"));
            track.vstFxChain = readStringArray(getObjectProperty(trackVar, "vst_fx_chain"));
            track.vstFxBypassed = getBoolProperty(trackVar, "vst_fx_bypassed", false);
            track.vstFxSlotBypassed = normaliseBoolVector(readBoolVector(getObjectProperty(trackVar, "vst_fx_slot_bypassed")),
                                                          track.vstFxChain.size());
            track.midiProgram = getIntProperty(trackVar, "midi_program", 0, 0, 127);
            track.midiChannel = getIntProperty(trackVar, "midi_channel", trackIndex % 16, 0, 15);
            track.synthProfile = getStringProperty(trackVar, "synth_profile", track.synthProfile);
            track.renderedAudioPath = getStringProperty(trackVar, "rendered_audio_path");
            track.mute = getBoolProperty(trackVar, "mute", false);
            track.solo = getBoolProperty(trackVar, "solo", false);
            track.liveArmed = getBoolProperty(trackVar, "live_armed", false);
            track.colorHex = getStringProperty(trackVar, "color_hex");
            track.followThemeTrackColour = getBoolProperty(trackVar,
                                                           "theme_track_colour",
                                                           track.colorHex.trim().isEmpty() || isLegacyAutoTrackColour(track.colorHex));
            track.themeColourSlot = getIntProperty(trackVar, "theme_colour_slot", trackIndex, 0, 4096);
            track.arrangementVisibleAutomationTargets = readStringArray(getObjectProperty(trackVar,
                                                                                          "arrangement_visible_automation_targets"));
            track.routingTarget = getStringProperty(trackVar, "routing_target", "master");

            if (auto* notesArray = getObjectProperty(trackVar, "notes").getArray())
            {
                track.notes.reserve(static_cast<size_t>(notesArray->size()));
                for (const auto& noteVar : *notesArray)
                {
                    MidiNote note;
                    note.startTick = getIntProperty(noteVar, "start_tick", 0, 0);
                    note.durationTick = getIntProperty(noteVar, "duration_tick", kTicksPerBeat / 2, 1);
                    note.pitch = getIntProperty(noteVar, "pitch", 60, 0, 127);
                    note.velocity = getIntProperty(noteVar, "velocity", 100, 1, 127);
                    note.selected = getBoolProperty(noteVar, "selected", false);
                    track.notes.push_back(note);
                }
            }

            readAutomationLanes(getObjectProperty(trackVar, "automation_lanes"), track.automationLanes);
            sanitiseAutomationLanes(track);
            sanitiseArrangementAutomationLaneVisibility(track);
            loaded.project.tracks.push_back(std::move(track));
        }
    }

    if (loaded.project.tracks.empty())
        loaded.project.tracks = makeDefaultTracks();

    loaded.project.vstiPaths = readStringArray(getObjectProperty(projectRoot, "vsti_paths"));
    loaded.project.vstiFolderPaths = readStringArray(getObjectProperty(projectRoot, "vsti_folder_paths"));
    loaded.project.samplePaths = readStringArray(getObjectProperty(projectRoot, "sample_paths"));
    loaded.project.masterVolume = getDoubleProperty(projectRoot, "master_volume", 1.0, 0.0, 2.0);
    loaded.project.masterFxChain = readStringArray(getObjectProperty(projectRoot, "master_fx_chain"));
    loaded.project.masterFxBypassed = getBoolProperty(projectRoot, "master_fx_bypassed", false);
    loaded.project.masterFxSlotBypassed = normaliseBoolVector(readBoolVector(getObjectProperty(projectRoot, "master_fx_slot_bypassed")),
                                                              loaded.project.masterFxChain.size());
    loaded.project.sharedFxBuses.clear();
    if (auto* sharedFxBusArray = getObjectProperty(projectRoot, "shared_fx_buses").getArray())
    {
        loaded.project.sharedFxBuses.reserve(static_cast<size_t>(sharedFxBusArray->size()));
        for (const auto& item : *sharedFxBusArray)
        {
            SharedEffectBusState bus;
            bus.id = getStringProperty(item, "id", juce::Uuid().toString());
            bus.name = getStringProperty(item, "name", bus.name);
            bus.effect = getStringProperty(item, "effect");
            if (item.hasProperty("output_targets"))
                bus.outputTargets = readStringArray(getObjectProperty(item, "output_targets"));
            else
                bus.outputTargets.add("master");
            readVstParameters(getObjectProperty(item, "parameters"), bus.parameters);
            bus.statePath = getStringProperty(item, "state_path");
            bus.bypassed = getBoolProperty(item, "bypassed", false);
            loaded.project.sharedFxBuses.push_back(std::move(bus));
        }
    }

    loaded.project.vstRack.clear();
    if (auto* rackArray = getObjectProperty(projectRoot, "vsti_rack").getArray())
    {
        loaded.project.vstRack.reserve(static_cast<size_t>(rackArray->size()));
        for (const auto& item : *rackArray)
        {
            VstInstrument instrument;
            instrument.name = getStringProperty(item, "name");
            instrument.path = getStringProperty(item, "path");
            instrument.pluginName = getStringProperty(item, "plugin_name");
            instrument.isInstrument = getBoolProperty(item, "is_instrument", false);
            instrument.isEffect = getBoolProperty(item, "is_effect", false);
            instrument.category = getStringProperty(item, "category");
            instrument.hostSupported = getBoolProperty(item, "host_supported", true);
            instrument.hostError = getStringProperty(item, "host_error");
            loaded.project.vstRack.push_back(std::move(instrument));
        }
    }

    loaded.project.sampleAssets.clear();
    if (auto* assetArray = getObjectProperty(projectRoot, "sample_assets").getArray())
    {
        loaded.project.sampleAssets.reserve(static_cast<size_t>(assetArray->size()));
        for (const auto& item : *assetArray)
        {
            SampleAsset asset;
            asset.path = getStringProperty(item, "path");
            asset.durationSec = getDoubleProperty(item, "duration_sec", 0.0, 0.0);
            asset.sampleRate = getIntProperty(item, "sample_rate", 44100, 1000, 384000);
            asset.waveformPreview = readFloatVector(getObjectProperty(item, "waveform_preview"));
            loaded.project.sampleAssets.push_back(std::move(asset));
        }
    }

    loaded.project.sampleClips.clear();
    if (auto* clipArray = getObjectProperty(projectRoot, "sample_clips").getArray())
    {
        loaded.project.sampleClips.reserve(static_cast<size_t>(clipArray->size()));
        for (const auto& item : *clipArray)
        {
            SampleClip clip;
            clip.path = getStringProperty(item, "path");
            clip.trackIndex = getIntProperty(item, "track_index", 0, 0);
            clip.startSec = getDoubleProperty(item, "start_sec", 0.0, 0.0);
            clip.durationSec = getDoubleProperty(item, "duration_sec", 0.0, 0.0);
            clip.sourceOffsetSec = getDoubleProperty(item, "source_offset_sec", 0.0, 0.0);
            clip.sourceFileDurationSec = getDoubleProperty(item,
                                                           "source_file_duration_sec",
                                                           clip.durationSec,
                                                           0.0);
            clip.sampleRate = getIntProperty(item, "sample_rate", 44100, 1000, 384000);
            clip.waveformPreview = readFloatVector(getObjectProperty(item, "waveform_preview"));
            if (clip.sourceFileDurationSec <= 0.0)
                clip.sourceFileDurationSec = clip.durationSec;
            loaded.project.sampleClips.push_back(std::move(clip));
        }
    }

    loaded.project.midiPatterns.clear();
    if (auto* patternArray = getObjectProperty(projectRoot, "midi_patterns").getArray())
    {
        loaded.project.midiPatterns.reserve(static_cast<size_t>(patternArray->size()));
        for (int patternIndex = 0; patternIndex < patternArray->size(); ++patternIndex)
        {
            const auto& item = patternArray->getReference(patternIndex);
            MidiPattern pattern;
            pattern.id = getStringProperty(item, "id", juce::Uuid().toString());
            pattern.name = getStringProperty(item, "name", "Pattern " + juce::String(patternIndex + 1));
            const auto legacyPatternBarsVar = getObjectProperty(item, "length_bars");
            const auto legacyPatternLengthTicks = legacyPatternBarsVar.isVoid()
                ? defaultPatternLengthTicks(loaded.project)
                : getIntProperty(item, "length_bars", 1, 1, 128) * ticksPerBar(loaded.project);
            pattern.lengthTicks = getIntProperty(item,
                                                 "length_ticks",
                                                 legacyPatternLengthTicks,
                                                 kMinSequenceSnapTicks);

            if (auto* notesArray = getObjectProperty(item, "notes").getArray())
            {
                pattern.notes.reserve(static_cast<size_t>(notesArray->size()));
                for (const auto& noteVar : *notesArray)
                {
                    MidiNote note;
                    note.startTick = getIntProperty(noteVar, "start_tick", 0, 0);
                    note.durationTick = getIntProperty(noteVar, "duration_tick", kTicksPerBeat / 2, 1);
                    note.pitch = getIntProperty(noteVar, "pitch", 60, 0, 127);
                    note.velocity = getIntProperty(noteVar, "velocity", 100, 1, 127);
                    note.selected = getBoolProperty(noteVar, "selected", false);
                    pattern.notes.push_back(note);
                }
            }

            readAutomationLanes(getObjectProperty(item, "controller_lanes"), pattern.controllerLanes);
            sanitiseMidiNoteList(pattern.notes);
            sanitisePatternControllerLanes(pattern);
            loaded.project.midiPatterns.push_back(std::move(pattern));
        }
    }

    loaded.project.midiSections.clear();
    if (auto* sectionArray = getObjectProperty(projectRoot, "midi_sections").getArray())
    {
        loaded.project.midiSections.reserve(static_cast<size_t>(sectionArray->size()));
        for (const auto& item : *sectionArray)
        {
            MidiSection section;
            section.trackIndex = getIntProperty(item, "track_index", 0, 0);
            section.startTick = getIntProperty(item, "start_tick",
                                               secondsToTick(getDoubleProperty(item, "start_sec", 0.0, 0.0), loaded.project.bpm),
                                               0);
            const auto legacySectionBarsVar = getObjectProperty(item, "length_bars");
            const auto legacySectionDurationTicks = juce::jmax(kMinSequenceSnapTicks,
                                                               secondsToTick(getDoubleProperty(item,
                                                                                               "duration_sec",
                                                                                               tickToSeconds(ticksPerBar(loaded.project), loaded.project.bpm),
                                                                                               0.0),
                                                                             loaded.project.bpm));
            const auto legacySectionLengthTicks = legacySectionBarsVar.isVoid()
                ? legacySectionDurationTicks
                : getIntProperty(item, "length_bars", 1, 1, 128) * ticksPerBar(loaded.project);
            section.lengthTicks = getIntProperty(item,
                                                 "length_ticks",
                                                 legacySectionLengthTicks,
                                                 kMinSequenceSnapTicks);
            section.startSec = getDoubleProperty(item, "start_sec", 0.0, 0.0);
            section.durationSec = getDoubleProperty(item, "duration_sec", 0.0, 0.0);
            section.name = getStringProperty(item, "name", section.name);
            section.patternId = getStringProperty(item, "pattern_id");
            loaded.project.midiSections.push_back(std::move(section));
        }
    }

    migrateLegacyTrackNotesToPatterns(loaded.project);
    synchronisePatternClipsToTrackNotes(loaded.project);

    loaded.project.recalculateTimeFields();
    outProject = std::move(loaded);
    return juce::Result::ok();
}

juce::Result saveProjectFile(const juce::File& file, const ProjectFileData& projectFile)
{
    if (file == juce::File())
        return juce::Result::fail("No destination file was provided.");

    ProjectState normalisedProject = projectFile.project;
    normalisedProject.recalculateTimeFields();
    for (auto& track : normalisedProject.tracks)
        sanitiseAutomationLanes(track);

    auto root = makeObjectVar();
    auto projectVar = makeObjectVar();
    juce::Array<juce::var> trackArray;
    juce::Array<juce::var> rackArray;
    juce::Array<juce::var> assetArray;
    juce::Array<juce::var> clipArray;
    juce::Array<juce::var> patternArray;
    juce::Array<juce::var> sectionArray;

    for (const auto& track : normalisedProject.tracks)
        trackArray.add(serialiseTrack(track));

    for (const auto& instrument : normalisedProject.vstRack)
        rackArray.add(serialiseVstInstrument(instrument));

    for (const auto& asset : normalisedProject.sampleAssets)
        assetArray.add(serialiseSampleAsset(asset));

    for (const auto& clip : normalisedProject.sampleClips)
        clipArray.add(serialiseSampleClip(clip));

    const auto projectBarTicks = ticksPerBar(normalisedProject);

    for (const auto& pattern : normalisedProject.midiPatterns)
        patternArray.add(serialiseMidiPattern(pattern, projectBarTicks));

    for (const auto& section : normalisedProject.midiSections)
        sectionArray.add(serialiseMidiSection(section, projectBarTicks));

    setObjectProperty(root, "format", projectFile.format.isNotEmpty() ? projectFile.format : "ai_music_studio_project");
    setObjectProperty(root, "version", juce::jmax(projectFile.version, kProjectFileVersion));
    setObjectProperty(root, "saved_at_unix", juce::Time::getCurrentTime().toMilliseconds() / 1000);

    setObjectProperty(projectVar, "bpm", normalisedProject.bpm);
    setObjectProperty(projectVar, "time_signature_numerator", normalisedProject.timeSigNumerator);
    setObjectProperty(projectVar, "time_signature_denominator", normalisedProject.timeSigDenominator);
    setObjectProperty(projectVar, "default_pattern_ticks", normalisedProject.defaultPatternTicks);
    setObjectProperty(projectVar, "default_pattern_bars", juce::jmax(1, (normalisedProject.defaultPatternTicks + projectBarTicks - 1) / projectBarTicks));
    setObjectProperty(projectVar, "quantize_enabled", normalisedProject.quantizeEnabled);
    setObjectProperty(projectVar, "quantize_div", normalisedProject.quantizeDiv);
    setObjectProperty(projectVar, "quantize_triplet", normalisedProject.quantizeTriplet);
    setObjectProperty(projectVar, "key_quantize_root", normalisedProject.keyQuantizeRoot);
    setObjectProperty(projectVar, "key_quantize_scale", normalisedProject.keyQuantizeScale);
    setObjectProperty(projectVar, "piano_roll_visible_pitch_min", normalisedProject.pianoRollVisiblePitchMin);
    setObjectProperty(projectVar, "piano_roll_visible_pitch_max", normalisedProject.pianoRollVisiblePitchMax);
    setObjectProperty(projectVar, "arrangement_snap_ticks", normalisedProject.arrangementSnapTicks);
    setObjectProperty(projectVar, "loop_enabled", normalisedProject.loopEnabled);
    setObjectProperty(projectVar, "metronome_enabled", normalisedProject.metronomeEnabled);
    setObjectProperty(projectVar, "left_locator_tick", normalisedProject.leftLocatorTick);
    setObjectProperty(projectVar, "right_locator_tick", normalisedProject.rightLocatorTick);
    setObjectProperty(projectVar, "playhead_tick", normalisedProject.playheadTick);
    setObjectProperty(projectVar, "left_locator_sec", normalisedProject.leftLocatorSec);
    setObjectProperty(projectVar, "right_locator_sec", normalisedProject.rightLocatorSec);
    setObjectProperty(projectVar, "playhead_sec", normalisedProject.playheadSec);
    setObjectProperty(projectVar, "master_volume", normalisedProject.masterVolume);
    setObjectProperty(projectVar, "master_fx_chain", makeStringArrayVar(normalisedProject.masterFxChain));
    setObjectProperty(projectVar, "master_fx_bypassed", normalisedProject.masterFxBypassed);
    setObjectProperty(projectVar, "master_fx_slot_bypassed",
                      makeBoolArrayVar(normalisedProject.masterFxSlotBypassed, normalisedProject.masterFxChain.size()));
    {
        juce::Array<juce::var> sharedFxBusArray;
        for (const auto& bus : normalisedProject.sharedFxBuses)
            sharedFxBusArray.add(serialiseSharedEffectBus(bus));
        setObjectProperty(projectVar, "shared_fx_buses", juce::var(sharedFxBusArray));
    }
    setObjectProperty(projectVar, "tracks", juce::var(trackArray));
    setObjectProperty(projectVar, "vsti_paths", makeStringArrayVar(normalisedProject.vstiPaths));
    setObjectProperty(projectVar, "vsti_folder_paths", makeStringArrayVar(normalisedProject.vstiFolderPaths));
    setObjectProperty(projectVar, "sample_paths", makeStringArrayVar(normalisedProject.samplePaths));
    setObjectProperty(projectVar, "vsti_rack", juce::var(rackArray));
    setObjectProperty(projectVar, "sample_assets", juce::var(assetArray));
    setObjectProperty(projectVar, "sample_clips", juce::var(clipArray));
    setObjectProperty(projectVar, "midi_patterns", juce::var(patternArray));
    setObjectProperty(projectVar, "midi_sections", juce::var(sectionArray));
    setObjectProperty(root, "project", projectVar);
    setObjectProperty(root, "ui", projectFile.uiState.isVoid() ? makeObjectVar() : projectFile.uiState);

    const auto parent = file.getParentDirectory();
    if (!parent.createDirectory())
        return juce::Result::fail("Failed to create project folder: " + parent.getFullPathName());

    if (!file.replaceWithText(juce::JSON::toString(root, false), false, false, "\n"))
        return juce::Result::fail("Failed to write project file.");

    return juce::Result::ok();
}

} // namespace aims
