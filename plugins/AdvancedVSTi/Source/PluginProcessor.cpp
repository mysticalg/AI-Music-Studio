#include "PluginProcessor.h"
#include "PluginEditor.h"

#if JUCE_WINDOWS
#include <windows.h>
#endif

namespace
{
constexpr float twoPi = juce::MathConstants<float>::twoPi;
constexpr int pluginStateVersion = 3;
constexpr std::array<const char*, 15> drumVoiceLevelParamIds {
    "DRUMLEVEL_KICK",
    "DRUMLEVEL_SNARE",
    "DRUMLEVEL_RIM",
    "DRUMLEVEL_CLAP",
    "DRUMLEVEL_CLOSED_HAT",
    "DRUMLEVEL_OPEN_HAT",
    "DRUMLEVEL_LOW_TOM",
    "DRUMLEVEL_MID_TOM",
    "DRUMLEVEL_HIGH_TOM",
    "DRUMLEVEL_CRASH",
    "DRUMLEVEL_RIDE",
    "DRUMLEVEL_COWBELL",
    "DRUMLEVEL_CLAVE",
    "DRUMLEVEL_MARACA",
    "DRUMLEVEL_PERC"
};

constexpr std::array<const char*, 15> drumVoiceLevelNames {
    "Bass Drum Level",
    "Snare Level",
    "Rim Shot Level",
    "Clap Level",
    "Closed Hat Level",
    "Open Hat Level",
    "Low Tom Level",
    "Mid Tom Level",
    "High Tom Level",
    "Crash Level",
    "Ride Level",
    "Cowbell Level",
    "Clave Level",
    "Maraca Level",
    "Perc Level"
};

constexpr std::array<float, 15> drumMachineVoiceLevelDefaults {
    1.0f, 0.96f, 0.72f, 0.9f, 0.84f,
    0.88f, 0.84f, 0.82f, 0.8f, 0.78f,
    0.76f, 0.68f, 0.66f, 0.62f, 0.72f
};

constexpr std::array<float, 15> drum808VoiceLevelDefaults {
    1.08f, 0.94f, 0.7f, 0.8f, 0.72f,
    0.8f, 0.88f, 0.84f, 0.8f, 0.72f,
    0.74f, 0.68f, 0.64f, 0.66f, 0.72f
};

constexpr int vecPadMidiStart = 36;
constexpr int maxVecPadSampleChoices = 4096;
constexpr std::array<const char*, 23> vecPadFolderOrder {
    "VEC1 Bassdrums",
    "VEC1 Bassdrums",
    "VEC1 Snares",
    "VEC1 Snares",
    "VEC1 Claps",
    "VEC1 Claps",
    "VEC1 Cymbals",
    "VEC1 Cymbals",
    "VEC1 Cymbals",
    "VEC1 Cymbals",
    "VEC1 Cymbals",
    "VEC1 Percussion",
    "VEC1 FX",
    "VEC1 Fills",
    "VEC1 BreakBeats",
    "VEC1 303 Acid",
    "VEC1 Long Basses",
    "VEC1 Offbeat Bass",
    "VEC1 Loops",
    "VEC1 Multis",
    "VEC1 Sounds",
    "VEC1 Special Sounds From Produc",
    "VEC1 Vinyl FX and Scratches"
};

constexpr std::array<const char*, 23> vecPadTitles {
    "Bassdrum C",
    "Bassdrum C#",
    "Snare D",
    "Snare D#",
    "Clap E",
    "Clap F",
    "Closed HH",
    "Open HH",
    "Ride",
    "Crash",
    "Rev Crash",
    "Perc",
    "FX",
    "Fills",
    "Breaks",
    "303 Acid",
    "Long Bass",
    "Offbeat",
    "Loops",
    "Multis",
    "Sounds",
    "Special",
    "Vinyl FX"
};

constexpr std::array<const char*, 23> vecPadNotes {
    "C1",  "C#1", "D1",  "D#1", "E1",  "F1",
    "F#1", "G1",  "G#1", "A1",  "A#1", "B1",
    "C2",  "C#2", "D2",  "D#2", "E2",  "F2",
    "F#2", "G2",  "G#2", "A2",  "A#2"
};

constexpr std::array<const char*, 23> vecPadSubfolderNames {
    "",
    "",
    "",
    "",
    "",
    "",
    "VEC1 Close HH",
    "VEC1 Open HH",
    "VEC1 Ride",
    "VEC1 Crash",
    "VEC1 Reverse Crash",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    ""
};

juce::StringArray classicOscillatorChoices()
{
    return { "Sine", "Saw", "Square", "Noise", "Sample" };
}

juce::StringArray advancedOscillatorChoices()
{
    return {
        "Sine",
        "Saw",
        "Square",
        "Noise",
        "Sample",
        "HyperSaw",
        "WT Formant",
        "WT Complex",
        "WT Metal",
        "WT Vocal"
    };
}

juce::StringArray builtInAdvancedVirusPresetChoices()
{
    return {
        "Init Saw", "Init Square", "Wide Pad", "Mono Lead",
        "TF Rectify Cm", "Feedy Seq Cm", "Rhy Arp", "Solar Systems Seq Cm",
        "Zipp Cm", "Squib Fm", "Infinity Dbm", "Rev UFO Cm",
        "Lazer Gallop Bbm", "Deep Solar Bass Cm", "Wavepad Cm", "Tweeker",
        "Poly Sin Fm", "Omni Cm", "Mystery G", "Big Sweep Cm"
    };
}

struct ImportedVirusPresetData
{
    juce::String name;
    juce::String bankLabel;
    juce::String sourceFile;
    int slot = 0;
    std::vector<uint8_t> payload;
};

constexpr std::array<uint8_t, 3> accessManufacturerId { 0x00, 0x20, 0x33 };
constexpr int virusPatchNameOffset = 248;
constexpr int virusPatchNameLength = 10;

juce::String decodeVirusPatchName (const std::vector<uint8_t>& payload)
{
    juce::String result;

    for (int index = 0; index < virusPatchNameLength; ++index)
    {
        const auto offset = virusPatchNameOffset + index;
        if (! juce::isPositiveAndBelow (offset, static_cast<int> (payload.size())))
            break;

        const auto byte = payload[static_cast<size_t> (offset)];
        result += (byte >= 32 && byte <= 126)
                      ? juce::String::charToString (static_cast<juce::juce_wchar> (byte))
                      : " ";
    }

    return result.trim();
}

juce::String shortVirusBankLabel (const juce::File& midiFile, int fileOrdinal)
{
    auto base = midiFile.getFileNameWithoutExtension().trim();
    base = base.retainCharacters ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ");

    juce::StringArray tokens;
    tokens.addTokens (base, " ", {});
    tokens.removeEmptyStrings();

    juce::String label;
    for (const auto& token : tokens)
    {
        if (token.containsOnly ("0123456789"))
            label << token;
        else
            label << token.substring (0, 1).toUpperCase();
    }

    if (label.isEmpty())
        label = "BANK" + juce::String (fileOrdinal + 1);

    return label.substring (0, 6);
}

std::vector<juce::File> findVirusBankMidiFiles()
{
    std::vector<juce::File> midiFiles;

    auto addMidiFilesFrom = [&midiFiles] (const juce::File& folder)
    {
        if (! folder.isDirectory())
            return;

        auto files = folder.findChildFiles (juce::File::findFiles, false, "*.mid");
        for (const auto& file : files)
        {
            if (! file.getFullPathName().containsIgnoreCase ("__MACOSX"))
                midiFiles.push_back (file);
        }
    };

    auto searchUpwardForFolder = [&addMidiFilesFrom] (juce::File start)
    {
        for (int depth = 0; depth < 12 && start.exists(); ++depth)
        {
            const auto candidate = start.getChildFile ("Virus_TI_EDM_Soundbank_2015");
            if (candidate.isDirectory())
            {
                addMidiFilesFrom (candidate);
                return true;
            }

            start = start.getParentDirectory();
        }

        return false;
    };

    searchUpwardForFolder (juce::File::getCurrentWorkingDirectory());
    searchUpwardForFolder (juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory());

   #if JUCE_WINDOWS
    HMODULE moduleHandle = nullptr;
    if (GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR> (&findVirusBankMidiFiles),
                            &moduleHandle) != 0)
    {
        wchar_t modulePath[MAX_PATH] {};
        if (GetModuleFileNameW (moduleHandle, modulePath, MAX_PATH) > 0)
            searchUpwardForFolder (juce::File (juce::String (modulePath)).getParentDirectory());
    }
   #endif

    std::sort (midiFiles.begin(), midiFiles.end(),
               [] (const juce::File& lhs, const juce::File& rhs)
               {
                   return lhs.getFullPathName().compareIgnoreCase (rhs.getFullPathName()) < 0;
               });

    midiFiles.erase (std::unique (midiFiles.begin(), midiFiles.end(),
                                  [] (const juce::File& lhs, const juce::File& rhs)
                                  {
                                      return lhs.getFullPathName().equalsIgnoreCase (rhs.getFullPathName());
                                  }),
                     midiFiles.end());
    return midiFiles;
}

std::vector<ImportedVirusPresetData> loadImportedVirusPresets()
{
    std::vector<ImportedVirusPresetData> presets;
    const auto midiFiles = findVirusBankMidiFiles();
    juce::StringArray usedNames;

    for (int fileIndex = 0; fileIndex < static_cast<int> (midiFiles.size()); ++fileIndex)
    {
        const auto& midiFile = midiFiles[static_cast<size_t> (fileIndex)];
        juce::FileInputStream stream (midiFile);
        if (! stream.openedOk())
            continue;

        juce::MidiFile midi;
        if (! midi.readFrom (stream))
            continue;

        const auto bankLabel = shortVirusBankLabel (midiFile, fileIndex);

        for (int trackIndex = 0; trackIndex < midi.getNumTracks(); ++trackIndex)
        {
            const auto* track = midi.getTrack (trackIndex);
            if (track == nullptr)
                continue;

            for (int eventIndex = 0; eventIndex < track->getNumEvents(); ++eventIndex)
            {
                const auto* event = track->getEventPointer (eventIndex);
                if (event == nullptr || ! event->message.isSysEx())
                    continue;

                const auto* data = event->message.getSysExData();
                const auto dataSize = event->message.getSysExDataSize();
                if (data == nullptr || dataSize < virusPatchNameOffset + virusPatchNameLength)
                    continue;
                if (static_cast<uint8_t> (data[0]) != accessManufacturerId[0]
                    || static_cast<uint8_t> (data[1]) != accessManufacturerId[1]
                    || static_cast<uint8_t> (data[2]) != accessManufacturerId[2])
                    continue;

                ImportedVirusPresetData preset;
                preset.bankLabel = bankLabel;
                preset.sourceFile = midiFile.getFileName();
                preset.slot = static_cast<uint8_t> (data[7]);
                preset.payload.assign (data, data + dataSize);

                auto patchName = decodeVirusPatchName (preset.payload);
                if (patchName.isEmpty())
                    patchName = "Imported " + juce::String (presets.size() + 1);

                auto uniqueName = patchName;
                int duplicateIndex = 2;
                while (usedNames.contains (uniqueName, true))
                    uniqueName = patchName + " (" + bankLabel + " " + juce::String (duplicateIndex++) + ")";

                preset.name = uniqueName;
                usedNames.add (uniqueName);
                presets.push_back (std::move (preset));
            }
        }
    }

    return presets;
}

const std::vector<ImportedVirusPresetData>& importedVirusPresets()
{
    static const auto presets = loadImportedVirusPresets();
    return presets;
}

float importedPayloadNorm (const ImportedVirusPresetData& preset, int offset)
{
    if (preset.payload.empty())
        return 0.0f;

    const auto index = static_cast<size_t> (juce::jlimit (0, static_cast<int> (preset.payload.size()) - 1, offset));
    return static_cast<float> (preset.payload[index]) / 127.0f;
}

bool importedNameContains (const juce::String& lowerName, std::initializer_list<const char*> needles)
{
    for (const auto* needle : needles)
    {
        if (lowerName.containsIgnoreCase (needle))
            return true;
    }

    return false;
}

float importedRange (const ImportedVirusPresetData& preset, int offset, float minValue, float maxValue)
{
    return juce::jmap (importedPayloadNorm (preset, offset), minValue, maxValue);
}

juce::StringArray virusLfoDestinationChoices()
{
    return {
        "OSC 1",
        "OSC 2/3",
        "PW",
        "RESO",
        "FLT GAIN",
        "CUTOFF 1",
        "CUTOFF 2",
        "SHAPE",
        "FM AMT",
        "PAN",
        "ASSIGN"
    };
}

juce::StringArray matrixSourceChoices()
{
    return {
        "Off",
        "LFO 1",
        "LFO 2",
        "LFO 3",
        "Filter Env",
        "Amp Env",
        "Velocity",
        "Note",
        "Random"
    };
}

juce::StringArray matrixDestinationChoices()
{
    return {
        "Off",
        "Osc1 Pitch",
        "Osc2/3 Pitch",
        "Pulse Width",
        "Resonance",
        "Filter Gain",
        "Cutoff 1",
        "Cutoff 2",
        "Shape",
        "FM Amount",
        "Panorama",
        "Assign",
        "Amp Level",
        "Filter Balance",
        "Osc Volume",
        "Sub Osc Volume",
        "Noise Volume",
        "FX Mix",
        "FX Intensity",
        "Delay Send",
        "Delay Time",
        "Delay Feedback",
        "Reverb Mix",
        "Reverb Time",
        "Low EQ Gain",
        "Mid EQ Gain",
        "High EQ Gain",
        "Reverb Damping",
        "Low EQ Freq",
        "Low EQ Q",
        "Mid EQ Freq",
        "Mid EQ Q",
        "High EQ Freq",
        "High EQ Q",
        "Master Level",
        "Detune",
        "Sync Amount",
        "Gate Length",
        "Filter Env Amt",
        "Osc2 Mix",
        "Ring Mod"
    };
}

struct WavetableFrameDescriptor
{
    std::array<float, 8> harmonics {};
    float phaseWarp = 0.0f;
    float asymmetry = 0.0f;
    float fold = 0.0f;
};

using WavetableFrameSet = std::array<WavetableFrameDescriptor, 5>;

constexpr WavetableFrameSet kVirusFormantFrames { {
    { { 1.00f, 0.24f, 0.12f, 0.08f, 0.04f, 0.02f, 0.00f, 0.00f },  0.02f,  0.01f, 0.02f },
    { { 0.92f, 0.42f, 0.26f, 0.12f, 0.08f, 0.04f, 0.01f, 0.00f },  0.05f,  0.04f, 0.06f },
    { { 0.78f, 0.58f, 0.36f, 0.20f, 0.12f, 0.06f, 0.03f, 0.00f },  0.08f,  0.06f, 0.10f },
    { { 0.62f, 0.46f, 0.52f, 0.30f, 0.18f, 0.10f, 0.06f, 0.02f },  0.11f,  0.10f, 0.14f },
    { { 0.44f, 0.34f, 0.56f, 0.42f, 0.24f, 0.16f, 0.10f, 0.04f },  0.14f,  0.12f, 0.18f }
} };

constexpr WavetableFrameSet kVirusComplexFrames { {
    { { 1.00f, 0.58f, 0.34f, 0.18f, 0.10f, 0.06f, 0.03f, 0.01f },  0.02f, -0.04f, 0.04f },
    { { 0.94f, 0.52f, 0.44f, 0.26f, 0.14f, 0.08f, 0.05f, 0.02f },  0.05f, -0.08f, 0.08f },
    { { 0.86f, 0.46f, 0.40f, 0.34f, 0.22f, 0.12f, 0.07f, 0.03f },  0.08f, -0.12f, 0.12f },
    { { 0.74f, 0.40f, 0.34f, 0.30f, 0.28f, 0.18f, 0.10f, 0.05f },  0.11f, -0.16f, 0.18f },
    { { 0.62f, 0.34f, 0.28f, 0.26f, 0.26f, 0.22f, 0.14f, 0.08f },  0.14f, -0.20f, 0.24f }
} };

constexpr WavetableFrameSet kVirusMetalFrames { {
    { { 1.00f, 0.18f, 0.42f, 0.10f, 0.26f, 0.08f, 0.18f, 0.04f },  0.06f,  0.08f, 0.10f },
    { { 0.92f, 0.14f, 0.48f, 0.16f, 0.32f, 0.12f, 0.24f, 0.06f },  0.10f,  0.12f, 0.18f },
    { { 0.84f, 0.10f, 0.54f, 0.22f, 0.36f, 0.18f, 0.28f, 0.08f },  0.14f,  0.16f, 0.28f },
    { { 0.72f, 0.08f, 0.48f, 0.28f, 0.40f, 0.24f, 0.34f, 0.12f },  0.18f,  0.20f, 0.40f },
    { { 0.60f, 0.06f, 0.40f, 0.34f, 0.44f, 0.30f, 0.38f, 0.16f },  0.22f,  0.24f, 0.54f }
} };

constexpr WavetableFrameSet kVirusVocalFrames { {
    { { 1.00f, 0.22f, 0.08f, 0.22f, 0.06f, 0.03f, 0.00f, 0.00f },  0.02f,  0.02f, 0.02f },
    { { 0.94f, 0.34f, 0.12f, 0.18f, 0.10f, 0.04f, 0.02f, 0.00f },  0.04f,  0.04f, 0.04f },
    { { 0.88f, 0.12f, 0.30f, 0.08f, 0.20f, 0.06f, 0.03f, 0.01f },  0.06f,  0.06f, 0.08f },
    { { 0.82f, 0.26f, 0.10f, 0.24f, 0.12f, 0.08f, 0.03f, 0.02f },  0.08f,  0.08f, 0.10f },
    { { 0.76f, 0.18f, 0.06f, 0.14f, 0.08f, 0.04f, 0.02f, 0.00f },  0.10f,  0.10f, 0.12f }
} };

const WavetableFrameSet& virusWavetableFramesForVariant (int variant)
{
    switch (variant)
    {
        case 0: return kVirusFormantFrames;
        case 1: return kVirusComplexFrames;
        case 2: return kVirusMetalFrames;
        case 3:
        default: return kVirusVocalFrames;
    }
}

float sampleVirusWavetableFrame (float phase,
                                 float frequency,
                                 double sampleRate,
                                 const WavetableFrameDescriptor& frame)
{
    const auto brightnessLimit = juce::jlimit (0.18f, 1.0f,
                                               static_cast<float> ((sampleRate * 0.42) / juce::jmax (frequency * 8.0f, 1.0f)));
    auto warpedPhase = phase + std::sin (twoPi * phase) * frame.phaseWarp * 0.1f;
    warpedPhase += std::sin (twoPi * phase * 2.0f) * frame.asymmetry * 0.025f;
    warpedPhase -= std::floor (warpedPhase);

    float sample = 0.0f;
    float normaliser = 0.0f;

    for (size_t harmonicIndex = 0; harmonicIndex < frame.harmonics.size(); ++harmonicIndex)
    {
        const auto harmonic = static_cast<float> (harmonicIndex + 1);
        if (frequency * harmonic >= static_cast<float> (sampleRate) * 0.45f)
            continue;

        const auto gain = frame.harmonics[harmonicIndex] * brightnessLimit;
        const auto phaseOffset = ((harmonicIndex % 2 == 0) ? frame.asymmetry : -frame.asymmetry) * 0.055f * harmonic;
        sample += std::sin (twoPi * ((warpedPhase * harmonic) + phaseOffset)) * gain;
        normaliser += std::abs (gain);
    }

    if (normaliser > 0.0001f)
        sample /= normaliser;

    const auto folded = std::tanh (sample * (1.0f + frame.fold * 2.8f));
    return juce::jlimit (-1.0f, 1.0f, juce::jmap (juce::jlimit (0.0f, 1.0f, frame.fold * 0.55f), sample, folded));
}

juce::File vecPadLibraryRoot()
{
   #if JUCE_WINDOWS
    HMODULE moduleHandle = nullptr;
    if (GetModuleHandleExW (GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR> (&vecPadLibraryRoot),
                            &moduleHandle) != 0)
    {
        wchar_t modulePath[MAX_PATH] {};
        if (GetModuleFileNameW (moduleHandle, modulePath, MAX_PATH) > 0)
        {
            auto moduleFile = juce::File (juce::String (modulePath));
            auto contentsDir = moduleFile.getParentDirectory();
            if (contentsDir.getFileName().equalsIgnoreCase ("x86_64-win"))
                contentsDir = contentsDir.getParentDirectory();

            if (contentsDir.getFileName().equalsIgnoreCase ("Contents"))
            {
                const auto bundledRoot = contentsDir.getChildFile ("Resources").getChildFile ("VEC1");
                if (bundledRoot.isDirectory())
                    return bundledRoot;

                const auto legacyBundledRoot = contentsDir.getParentDirectory().getChildFile ("Resources").getChildFile ("VEC1");
                if (legacyBundledRoot.isDirectory())
                    return legacyBundledRoot;
            }
        }
    }
   #endif

    return juce::File::createFileWithoutCheckingPath (R"(D:\OneDrive\Music\Sound Design\Sample Library\sample library\VEC1)");
}

bool isSupportedExternalSampleFile (const juce::File& file)
{
    const auto extension = file.getFileExtension().toLowerCase();
    return extension == ".wav" || extension == ".aif" || extension == ".aiff" || extension == ".flac" || extension == ".ogg";
}

juce::String cleanedExternalSampleName (const juce::File& file)
{
    auto name = file.getFileNameWithoutExtension();
    if (name.startsWithIgnoreCase ("VEC1 "))
        name = name.fromFirstOccurrenceOf (" ", false, false).trim();
    return name;
}

std::vector<juce::File> findSortedChildDirectories (const juce::File& folder)
{
    auto directories = folder.findChildFiles (juce::File::findDirectories, false);
    std::vector<juce::File> result;
    result.reserve (static_cast<size_t> (directories.size()));

    for (const auto& directory : directories)
    {
        const auto name = directory.getFileName();
        if (name.startsWithChar ('.') || name.startsWithIgnoreCase ("__") || name.containsIgnoreCase ("exs"))
            continue;

        result.push_back (directory);
    }

    std::sort (result.begin(), result.end(),
               [] (const juce::File& lhs, const juce::File& rhs)
               {
                   return lhs.getFileName().compareIgnoreCase (rhs.getFileName()) < 0;
               });
    return result;
}

std::vector<juce::File> findSortedAudioFiles (const juce::File& folder)
{
    auto files = folder.findChildFiles (juce::File::findFiles, false);
    std::vector<juce::File> result;
    result.reserve (static_cast<size_t> (files.size()));

    for (const auto& file : files)
    {
        if (isSupportedExternalSampleFile (file))
            result.push_back (file);
    }

    std::sort (result.begin(), result.end(),
               [] (const juce::File& lhs, const juce::File& rhs)
               {
                   return lhs.getFileName().compareIgnoreCase (rhs.getFileName()) < 0;
               });
    return result;
}

float paramValue (const juce::AudioProcessorValueTreeState& apvts, const char* paramId)
{
    if (const auto* value = apvts.getRawParameterValue (paramId))
        return value->load();

    jassertfalse;
    return 0.0f;
}

float paramValue (const juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId)
{
    return paramValue (apvts, paramId.toRawUTF8());
}

int paramIndex (const juce::AudioProcessorValueTreeState& apvts, const char* paramId)
{
    return juce::roundToInt (paramValue (apvts, paramId));
}

int paramIndex (const juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId)
{
    return paramIndex (apvts, paramId.toRawUTF8());
}

enum class DrumVoiceKind
{
    kick,
    snare,
    rim,
    clap,
    closedHat,
    openHat,
    lowTom,
    midTom,
    highTom,
    crash,
    ride,
    cowbell,
    clave,
    maraca,
    perc
};

float midiToHz (int note)
{
    return 440.0f * std::pow (2.0f, (note - 69) / 12.0f);
}

float shapedEnv (float value, float curve)
{
    if (curve >= 0.0f)
        return std::pow (juce::jlimit (0.0001f, 1.0f, value), 1.0f + 5.0f * curve);

    return 1.0f - std::pow (juce::jlimit (0.0f, 1.0f, 1.0f - value), 1.0f + 5.0f * -curve);
}

float softSaturate (float value)
{
    return std::tanh (value);
}

float smoothTowards (float target, float coeff, float& state)
{
    state += coeff * (target - state);
    return state;
}

DrumVoiceKind drumKindForMidi (int midiNote)
{
    // Bundled drum machines use a keyboard-friendly layout starting at C:
    // C kick, C# rim, D snare, D# clap, E closed hat, F open hat,
    // F# low tom, G mid tom, G# high tom, A crash, A# ride, B cowbell,
    // next C clave, C# maraca, D perc.
    if (midiNote >= 36)
    {
        switch (midiNote - 36)
        {
            case 0:  return DrumVoiceKind::kick;
            case 1:  return DrumVoiceKind::rim;
            case 2:  return DrumVoiceKind::snare;
            case 3:  return DrumVoiceKind::clap;
            case 4:  return DrumVoiceKind::closedHat;
            case 5:  return DrumVoiceKind::openHat;
            case 6:  return DrumVoiceKind::lowTom;
            case 7:  return DrumVoiceKind::midTom;
            case 8:  return DrumVoiceKind::highTom;
            case 9:  return DrumVoiceKind::crash;
            case 10: return DrumVoiceKind::ride;
            case 11: return DrumVoiceKind::cowbell;
            case 12: return DrumVoiceKind::clave;
            case 13: return DrumVoiceKind::maraca;
            case 14: return DrumVoiceKind::perc;
            default: break;
        }
    }

    switch (midiNote)
    {
        case 35: return DrumVoiceKind::kick;
        case 51:
        case 53:
        case 59: return DrumVoiceKind::ride;
        case 52:
        case 55:
        case 57: return DrumVoiceKind::crash;
        case 56: return DrumVoiceKind::cowbell;
        case 69:
        case 70:
        case 82: return DrumVoiceKind::maraca;
        case 75:
        case 76: return DrumVoiceKind::clave;
        default: return DrumVoiceKind::perc;
    }
}

bool isHatKind (DrumVoiceKind kind) noexcept
{
    return kind == DrumVoiceKind::closedHat || kind == DrumVoiceKind::openHat;
}

float normalizedLogValue (float value, float minValue, float maxValue) noexcept
{
    const auto safeValue = juce::jlimit (minValue, maxValue, value);
    const auto minLog = std::log (minValue);
    const auto maxLog = std::log (maxValue);
    return juce::jlimit (0.0f, 1.0f, (std::log (safeValue) - minLog) / (maxLog - minLog));
}

float filterCascadeBlendForSlope (int slopeIndex) noexcept
{
    switch (slopeIndex)
    {
        case 1: return 1.0f / 3.0f; // Approximate a 16 dB roll-off between 12 and 24 dB.
        case 2: return 1.0f;
        default: return 0.0f;
    }
}

float drumNoiseWeight (DrumVoiceKind kind) noexcept
{
    switch (kind)
    {
        case DrumVoiceKind::snare:
        case DrumVoiceKind::clap:
        case DrumVoiceKind::closedHat:
        case DrumVoiceKind::openHat:
        case DrumVoiceKind::crash:
        case DrumVoiceKind::ride:
        case DrumVoiceKind::maraca:
            return 0.26f;
        case DrumVoiceKind::rim:
        case DrumVoiceKind::cowbell:
        case DrumVoiceKind::clave:
        case DrumVoiceKind::perc:
            return 0.14f;
        default:
            return 0.08f;
    }
}

float drumTransientWeight (DrumVoiceKind kind) noexcept
{
    switch (kind)
    {
        case DrumVoiceKind::kick:
        case DrumVoiceKind::snare:
        case DrumVoiceKind::rim:
        case DrumVoiceKind::clap:
            return 0.1f;
        case DrumVoiceKind::closedHat:
        case DrumVoiceKind::openHat:
        case DrumVoiceKind::crash:
        case DrumVoiceKind::ride:
            return 0.05f;
        default:
            return 0.035f;
    }
}

constexpr size_t drumVoiceIndex (DrumVoiceKind kind) noexcept
{
    return static_cast<size_t> (kind);
}

constexpr std::array<DrumVoiceKind, 7> drumVoiceTuneKinds {
    DrumVoiceKind::kick,
    DrumVoiceKind::snare,
    DrumVoiceKind::lowTom,
    DrumVoiceKind::midTom,
    DrumVoiceKind::highTom,
    DrumVoiceKind::crash,
    DrumVoiceKind::ride
};

constexpr std::array<const char*, 7> drumVoiceTuneParamIds {
    "DRUMTUNE_KICK",
    "DRUMTUNE_SNARE",
    "DRUMTUNE_LOW_TOM",
    "DRUMTUNE_MID_TOM",
    "DRUMTUNE_HIGH_TOM",
    "DRUMTUNE_CRASH",
    "DRUMTUNE_RIDE"
};

constexpr std::array<const char*, 7> drumVoiceTuneNames {
    "Bass Drum Tune",
    "Snare Tune",
    "Low Tom Tune",
    "Mid Tom Tune",
    "High Tom Tune",
    "Crash Tune",
    "Ride Tune"
};

constexpr std::array<DrumVoiceKind, 6> drumVoiceDecayKinds {
    DrumVoiceKind::kick,
    DrumVoiceKind::lowTom,
    DrumVoiceKind::midTom,
    DrumVoiceKind::highTom,
    DrumVoiceKind::closedHat,
    DrumVoiceKind::openHat
};

constexpr std::array<const char*, 6> drumVoiceDecayParamIds {
    "DRUMDECAY_KICK",
    "DRUMDECAY_LOW_TOM",
    "DRUMDECAY_MID_TOM",
    "DRUMDECAY_HIGH_TOM",
    "DRUMDECAY_CLOSED_HAT",
    "DRUMDECAY_OPEN_HAT"
};

constexpr std::array<const char*, 6> drumVoiceDecayNames {
    "Bass Drum Decay",
    "Low Tom Decay",
    "Mid Tom Decay",
    "High Tom Decay",
    "Closed Hat Decay",
    "Open Hat Decay"
};

constexpr std::array<float, 7> drumMachineVoiceTuneDefaults {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

constexpr std::array<float, 7> drum808VoiceTuneDefaults {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

constexpr std::array<float, 6> drumMachineVoiceDecayDefaults {
    0.52f, 0.5f, 0.5f, 0.48f, 0.32f, 0.58f
};

constexpr std::array<float, 6> drum808VoiceDecayDefaults {
    0.72f, 0.62f, 0.58f, 0.54f, 0.28f, 0.74f
};

bool migrateLegacyDrumFilterChoice (juce::ValueTree state)
{
    if (! state.isValid())
        return false;

    if (state.hasProperty ("id") && state["id"].toString() == "FILTERTYPE")
    {
        const auto currentValue = static_cast<float> (state.getProperty ("value", 0.0f));
        state.setProperty ("value", juce::jlimit (1.0f, 4.0f, currentValue + 1.0f), nullptr);
        return true;
    }

    if (state.hasProperty ("FILTERTYPE"))
    {
        const auto currentValue = static_cast<float> (state.getProperty ("FILTERTYPE", 0.0f));
        state.setProperty ("FILTERTYPE", juce::jlimit (1.0f, 4.0f, currentValue + 1.0f), nullptr);
        return true;
    }

    for (int index = 0; index < state.getNumChildren(); ++index)
    {
        if (migrateLegacyDrumFilterChoice (state.getChild (index)))
            return true;
    }

    return false;
}
} // namespace

AdvancedVSTiAudioProcessor::AdvancedVSTiAudioProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMETERS", createParameterLayout())
{
    loadedSample.setSize (1, 1);
    loadedSample.clear();
    audioFormatManager.registerBasicFormats();
    initializeExternalPadLibrary();
    apvts.addParameterListener ("PRESET", this);
    if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
            apvts.addParameterListener (externalPadSampleParameterIdForIndex (padIndex), this);
    }
    currentProgramIndex = juce::jlimit (0, juce::jmax (0, presetChoicesForFlavor().size() - 1), paramIndex (apvts, "PRESET"));
}

AdvancedVSTiAudioProcessor::~AdvancedVSTiAudioProcessor()
{
    cancelPendingUpdate();
    if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
            apvts.removeParameterListener (externalPadSampleParameterIdForIndex (padIndex), this);
    }
    apvts.removeParameterListener ("PRESET", this);
}

void AdvancedVSTiAudioProcessor::previewDrumPad (int midiNote, float velocity)
{
    if constexpr (! isDrumFlavor())
        return;

    const juce::SpinLock::ScopedLockType lock (pendingPreviewMidiLock);
    pendingPreviewMidi.addEvent (juce::MidiMessage::noteOn (1,
                                                            juce::jlimit (0, 127, midiNote),
                                                            juce::jlimit (0.0f, 1.0f, velocity)),
                                 0);
}

bool AdvancedVSTiAudioProcessor::toggleArpHold()
{
    const bool enabled = ! arpHoldEnabled.load();
    arpHoldEnabled.store (enabled);
    if (! enabled)
        pendingReleaseHeldNotes.store (true);
    return enabled;
}

bool AdvancedVSTiAudioProcessor::isArpHoldEnabled() const noexcept
{
    return arpHoldEnabled.load();
}

void AdvancedVSTiAudioProcessor::panicAllNotes()
{
    pendingPanicAllNotes.store (true);
}

void AdvancedVSTiAudioProcessor::auditionPresetNote (int midiNote, float velocity, int durationMs)
{
    pendingAuditionVelocity.store (juce::jlimit (1, 127, juce::roundToInt (juce::jlimit (0.0f, 1.0f, velocity) * 127.0f)));
    pendingAuditionDurationMs.store (juce::jmax (80, durationMs));
    pendingAuditionNote.store (juce::jlimit (0, 127, midiNote));
}

bool AdvancedVSTiAudioProcessor::isVec1DrumPadFlavor() const noexcept
{
    return buildFlavor() == InstrumentFlavor::vec1DrumPad;
}

int AdvancedVSTiAudioProcessor::externalPadCount() const noexcept
{
    return isVec1DrumPadFlavor() ? vecPadCount : 0;
}

juce::String AdvancedVSTiAudioProcessor::externalPadLevelParameterId (int padIndex) const
{
    return externalPadLevelParameterIdForIndex (padIndex);
}

juce::String AdvancedVSTiAudioProcessor::externalPadSustainParameterId (int padIndex) const
{
    return externalPadSustainParameterIdForIndex (padIndex);
}

juce::String AdvancedVSTiAudioProcessor::externalPadReleaseParameterId (int padIndex) const
{
    return externalPadReleaseParameterIdForIndex (padIndex);
}

AdvancedVSTiAudioProcessor::ExternalPadState AdvancedVSTiAudioProcessor::getExternalPadState (int padIndex) const
{
    ExternalPadState state;
    if (! isVec1DrumPadFlavor() || ! juce::isPositiveAndBelow (padIndex, vecPadCount))
        return state;

    state.title = vecPadTitles[static_cast<size_t> (padIndex)];
    state.note = vecPadNotes[static_cast<size_t> (padIndex)];
    state.midiNote = vecPadMidiStart + padIndex;

    if (! juce::isPositiveAndBelow (padIndex, static_cast<int> (externalPads.size())))
    {
        state.sample = "Library missing";
        return state;
    }

    const auto& pad = externalPads[static_cast<size_t> (padIndex)];
    if (! pad.displayName.isEmpty())
        state.title = pad.displayName;
    if (! pad.noteName.isEmpty())
        state.note = pad.noteName;
    state.midiNote = pad.midiNote;

    if (pad.samples.empty())
    {
        state.sample = "No samples";
        return state;
    }

    const auto selectedIndex = juce::jlimit (0,
                                             juce::jmax (0, static_cast<int> (pad.samples.size()) - 1),
                                             paramIndex (apvts, externalPadSampleParameterIdForIndex (padIndex).toRawUTF8()));
    const auto& selectedSample = pad.samples[static_cast<size_t> (selectedIndex)];
    state.preset = selectedSample.presetName;
    state.sample = selectedSample.displayName;
    state.canStepLeft = selectedIndex > 0;
    state.canStepRight = selectedIndex + 1 < static_cast<int> (pad.samples.size());
    return state;
}

void AdvancedVSTiAudioProcessor::stepExternalPadSample (int padIndex, int delta)
{
    if (! isVec1DrumPadFlavor() || ! juce::isPositiveAndBelow (padIndex, static_cast<int> (externalPads.size())) || delta == 0)
        return;

    const auto& pad = externalPads[static_cast<size_t> (padIndex)];
    if (pad.samples.empty())
        return;

    const auto currentIndex = juce::jlimit (0,
                                            juce::jmax (0, static_cast<int> (pad.samples.size()) - 1),
                                            paramIndex (apvts, externalPadSampleParameterIdForIndex (padIndex).toRawUTF8()));
    const auto nextIndex = juce::jlimit (0, static_cast<int> (pad.samples.size()) - 1, currentIndex + delta);
    if (nextIndex == currentIndex)
        return;

    setParameterActual (externalPadSampleParameterIdForIndex (padIndex).toRawUTF8(), static_cast<float> (nextIndex));
}

void AdvancedVSTiAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    loadedSampleBank = -1;

    juce::dsp::ProcessSpec spec { sampleRate, static_cast<juce::uint32> (samplesPerBlock), 1 };
    leftFilter.prepare (spec);
    rightFilter.prepare (spec);
    leftFilterCascade.prepare (spec);
    rightFilterCascade.prepare (spec);
    leftFilter2.prepare (spec);
    rightFilter2.prepare (spec);
    leftFilter2Cascade.prepare (spec);
    rightFilter2Cascade.prepare (spec);
    leftFilter.reset();
    rightFilter.reset();
    leftFilterCascade.reset();
    rightFilterCascade.reset();
    leftFilter2.reset();
    rightFilter2.reset();
    leftFilter2Cascade.reset();
    rightFilter2Cascade.reset();

    chorusLeft.prepare (spec);
    chorusRight.prepare (spec);
    chorusLeft.reset();
    chorusRight.reset();
    flangerLeft.prepare (spec);
    flangerRight.prepare (spec);
    flangerLeft.reset();
    flangerRight.reset();
    phaserLeft.prepare (spec);
    phaserRight.prepare (spec);
    phaserLeft.reset();
    phaserRight.reset();
    lowEqLeft.prepare (spec);
    lowEqRight.prepare (spec);
    lowEqLeft.reset();
    lowEqRight.reset();
    midEqLeft.prepare (spec);
    midEqRight.prepare (spec);
    midEqLeft.reset();
    midEqRight.reset();
    highEqLeft.prepare (spec);
    highEqRight.prepare (spec);
    highEqLeft.reset();
    highEqRight.reset();
    reverb.reset();
    delayBuffer.setSize (2, juce::jmax (samplesPerBlock * 4, static_cast<int> (sampleRate * 3.0)));
    delayBuffer.clear();
    delayWritePosition = 0;

    for (auto& voice : voices)
    {
        voice.ampEnv.setSampleRate (sampleRate);
        voice.filterEnv.setSampleRate (sampleRate);
        voice.toneState = 0.0f;
        voice.colourState = 0.0f;
    }

    updateRenderParameters();
    refreshSampleBank();
    refreshExternalPadSamples();
    applyEnvelopeSettings();
    reset();
}

void AdvancedVSTiAudioProcessor::releaseResources() {}

void AdvancedVSTiAudioProcessor::reset()
{
    heldNotes.clear();
    arpStep = 0;
    lfo1Phase = 0.0f;
    lfo2Phase = 0.0f;
    lfo3Phase = 0.0f;
    arpHoldEnabled.store (false);
    pendingReleaseHeldNotes.store (false);
    pendingPanicAllNotes.store (false);
    pendingAuditionNote.store (-1);
    activeAuditionNote = -1;
    auditionSamplesRemaining = 0;

    leftFilter.reset();
    rightFilter.reset();
    leftFilterCascade.reset();
    rightFilterCascade.reset();
    leftFilter2.reset();
    rightFilter2.reset();
    leftFilter2Cascade.reset();
    rightFilter2Cascade.reset();
    chorusLeft.reset();
    chorusRight.reset();
    flangerLeft.reset();
    flangerRight.reset();
    phaserLeft.reset();
    phaserRight.reset();
    lowEqLeft.reset();
    lowEqRight.reset();
    midEqLeft.reset();
    midEqRight.reset();
    highEqLeft.reset();
    highEqRight.reset();
    reverb.reset();
    delayBuffer.clear();
    delayWritePosition = 0;
    currentFilterEnvPeak = 0.0f;

    for (auto& voice : voices)
    {
        voice.active = false;
        voice.midiNote = -1;
        voice.velocity = 0.0f;
        voice.phase = 0.0f;
        voice.osc2Phase = 0.0f;
        voice.subPhase = 0.0f;
        voice.fmPhase = 0.0f;
        voice.syncPhase = 0.0f;
        voice.samplePos = 0.0f;
        voice.noteAge = 0.0f;
        voice.auxPhase = 0.0f;
        voice.toneState = 0.0f;
        voice.colourState = 0.0f;
        voice.externalPadIndex = -1;
        voice.externalSamplePosition = 0.0;
        voice.externalSample.reset();
        voice.unisonPhases.fill (0.0f);
        voice.unisonSyncPhases.fill (0.0f);
        voice.unisonSamplePositions.fill (0.0f);
        voice.ampEnv.reset();
        voice.filterEnv.reset();
    }
}

juce::StringArray AdvancedVSTiAudioProcessor::sampleBankChoices()
{
    return { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" };
}

juce::StringArray AdvancedVSTiAudioProcessor::presetChoicesForFlavor()
{
    if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
        return { "Classic 909", "Tight Club", "Dusty Machine" };
    if constexpr (buildFlavor() == InstrumentFlavor::drum808)
        return { "Classic 808", "Deep 808", "Sharp Electro" };
    if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
        return { "VEC1 Default" };
    if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
        return { "Sub Bass", "Saw Bass", "Square Bass", "Picked Bass" };
    if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
        return {
            "Ensemble", "Soft Strings", "Synth Brass", "Warm Choir", "Cinema Swell", "Studio Section",
            "Vintage Solina", "Octave Ribbon", "Silk Pad", "Muted Chamber", "Airy Tremolo", "Dark Legato",
            "Glass Arcs", "Dream Cascade", "Wide Overture", "Rosin Lead", "Analog Sweep", "Halo Choir",
            "Frozen Bow", "Pulse Ensemble", "Detuned Wash", "Ribbon Machine", "Golden Swell", "Velvet Brass",
            "Aurora Choir", "Neon Strings", "Mono Cello", "Nordic Bloom", "Cathedral Pad", "Lush Motion"
        };
    if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
        return { "Saw Lead", "Square Solo", "Soft Lead", "Brass Lead" };
    if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
        return { "Warm Pad", "Analog Pad", "Choir Pad", "Slow Sweep" };
    if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
        return { "Poly Pluck", "Bell Pluck", "Muted Pluck", "Glass Pluck" };
    if constexpr (buildFlavor() == InstrumentFlavor::sampler)
        return { "Dusty Keys", "Tape Choir", "Velvet Pluck", "Vox Chop", "Sub Stab", "Glass Bell" };
    if constexpr (buildFlavor() == InstrumentFlavor::acid303)
        return { "Acid Saw", "Acid Square", "Rounded 303", "Reso Line" };

    auto names = builtInAdvancedVirusPresetChoices();
    for (const auto& imported : importedVirusPresets())
        names.add (imported.name);
    return names;
}

juce::String AdvancedVSTiAudioProcessor::externalPadSampleParameterIdForIndex (int padIndex)
{
    return "VECPADSELECT_" + juce::String (padIndex + 1);
}

juce::String AdvancedVSTiAudioProcessor::externalPadLevelParameterIdForIndex (int padIndex)
{
    return "VECPADLEVEL_" + juce::String (padIndex + 1);
}

juce::String AdvancedVSTiAudioProcessor::externalPadSustainParameterIdForIndex (int padIndex)
{
    return "VECPADSUSTAIN_" + juce::String (padIndex + 1);
}

juce::String AdvancedVSTiAudioProcessor::externalPadReleaseParameterIdForIndex (int padIndex)
{
    return "VECPADRELEASE_" + juce::String (padIndex + 1);
}

void AdvancedVSTiAudioProcessor::initializeExternalPadLibrary()
{
    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
        return;

    externalPads.clear();
    externalPads.reserve (vecPadCount);

    const auto libraryRoot = vecPadLibraryRoot();
    const auto topLevelFolders = findSortedChildDirectories (libraryRoot);

    for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
    {
        ExternalPadDefinition pad;
        pad.folderName = vecPadFolderOrder[static_cast<size_t> (padIndex)];
        pad.displayName = vecPadTitles[static_cast<size_t> (padIndex)];
        pad.noteName = vecPadNotes[static_cast<size_t> (padIndex)];
        pad.midiNote = vecPadMidiStart + padIndex;
        const auto preferredSubfolder = juce::String (vecPadSubfolderNames[static_cast<size_t> (padIndex)]).trim();

        auto folderIt = std::find_if (topLevelFolders.begin(), topLevelFolders.end(),
                                      [&pad] (const juce::File& candidate)
                                      {
                                          return candidate.getFileName().compareIgnoreCase (pad.folderName) == 0;
                                      });

        if (folderIt != topLevelFolders.end())
        {
            auto presetFolders = findSortedChildDirectories (*folderIt);

            if (preferredSubfolder.isNotEmpty())
            {
                auto presetIt = std::find_if (presetFolders.begin(), presetFolders.end(),
                                              [&preferredSubfolder] (const juce::File& candidate)
                                              {
                                                  return candidate.getFileName().compareIgnoreCase (preferredSubfolder) == 0;
                                              });

                if (presetIt != presetFolders.end())
                {
                    for (const auto& sampleFile : findSortedAudioFiles (*presetIt))
                    {
                        pad.samples.push_back ({
                            sampleFile.getFullPathName(),
                            cleanedExternalSampleName (sampleFile),
                            presetIt->getFileName().replace ("VEC1 ", "", true).trim()
                        });
                    }
                }
            }
            else
            {
                for (const auto& presetFolder : presetFolders)
                {
                    for (const auto& sampleFile : findSortedAudioFiles (presetFolder))
                    {
                        pad.samples.push_back ({
                            sampleFile.getFullPathName(),
                            cleanedExternalSampleName (sampleFile),
                            presetFolder.getFileName().replace ("VEC1 ", "", true).trim()
                        });
                    }
                }
            }

            if (pad.samples.empty())
            {
                for (const auto& sampleFile : findSortedAudioFiles (*folderIt))
                {
                    pad.samples.push_back ({
                        sampleFile.getFullPathName(),
                        cleanedExternalSampleName (sampleFile),
                        {}
                    });
                }
            }
        }

        externalPads.push_back (std::move (pad));
    }
}

void AdvancedVSTiAudioProcessor::refreshExternalPadSamples()
{
    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
        return;

    for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
        loadExternalPadSample (padIndex);
}

void AdvancedVSTiAudioProcessor::loadExternalPadSample (int padIndex)
{
    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
        return;

    if (! juce::isPositiveAndBelow (padIndex, static_cast<int> (externalPads.size())))
        return;

    const auto& pad = externalPads[static_cast<size_t> (padIndex)];
    if (pad.samples.empty())
    {
        std::atomic_store (&externalPadSamples[static_cast<size_t> (padIndex)], std::shared_ptr<const ExternalSampleData> {});
        loadedExternalPadIndices[static_cast<size_t> (padIndex)] = -1;
        return;
    }

    const auto desiredIndex = juce::jlimit (0,
                                            juce::jmax (0, static_cast<int> (pad.samples.size()) - 1),
                                            paramIndex (apvts, externalPadSampleParameterIdForIndex (padIndex).toRawUTF8()));
    if (loadedExternalPadIndices[static_cast<size_t> (padIndex)] == desiredIndex
        && std::atomic_load (&externalPadSamples[static_cast<size_t> (padIndex)]) != nullptr)
        return;

    const auto sourceFile = juce::File::createFileWithoutCheckingPath (pad.samples[static_cast<size_t> (desiredIndex)].filePath);
    std::unique_ptr<juce::AudioFormatReader> reader (audioFormatManager.createReaderFor (sourceFile));
    if (reader == nullptr || reader->lengthInSamples <= 0)
    {
        std::atomic_store (&externalPadSamples[static_cast<size_t> (padIndex)], std::shared_ptr<const ExternalSampleData> {});
        loadedExternalPadIndices[static_cast<size_t> (padIndex)] = -1;
        return;
    }

    auto sampleData = std::make_shared<ExternalSampleData>();
    sampleData->sampleRate = reader->sampleRate;
    sampleData->displayName = pad.samples[static_cast<size_t> (desiredIndex)].displayName;
    sampleData->presetName = pad.samples[static_cast<size_t> (desiredIndex)].presetName;
    sampleData->audio.setSize (1, static_cast<int> (reader->lengthInSamples));
    sampleData->audio.clear();

    juce::AudioBuffer<float> sourceBuffer (juce::jmax (1, static_cast<int> (reader->numChannels)),
                                           static_cast<int> (reader->lengthInSamples));
    reader->read (&sourceBuffer, 0, static_cast<int> (reader->lengthInSamples), 0, true, true);

    auto* destination = sampleData->audio.getWritePointer (0);
    for (int sample = 0; sample < sourceBuffer.getNumSamples(); ++sample)
    {
        float mixedSample = 0.0f;
        for (int channel = 0; channel < sourceBuffer.getNumChannels(); ++channel)
            mixedSample += sourceBuffer.getSample (channel, sample);

        destination[sample] = mixedSample / static_cast<float> (juce::jmax (1, sourceBuffer.getNumChannels()));
    }

    std::atomic_store (&externalPadSamples[static_cast<size_t> (padIndex)], std::shared_ptr<const ExternalSampleData> (sampleData));
    loadedExternalPadIndices[static_cast<size_t> (padIndex)] = desiredIndex;
}

int AdvancedVSTiAudioProcessor::externalPadIndexForMidi (int midiNote) const noexcept
{
    if (! isVec1DrumPadFlavor())
        return -1;

    const auto index = midiNote - vecPadMidiStart;
    return juce::isPositiveAndBelow (index, vecPadCount) ? index : -1;
}

juce::StringArray AdvancedVSTiAudioProcessor::presetNames() const
{
    return presetChoicesForFlavor();
}

int AdvancedVSTiAudioProcessor::getNumPrograms()
{
    return juce::jmax (1, presetChoicesForFlavor().size());
}

int AdvancedVSTiAudioProcessor::getCurrentProgram()
{
    return juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), currentProgramIndex);
}

void AdvancedVSTiAudioProcessor::setCurrentProgram (int index)
{
    auto* parameter = apvts.getParameter ("PRESET");
    if (parameter == nullptr)
        return;

    const auto clamped = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), index);
    currentProgramIndex = clamped;
    parameter->setValueNotifyingHost (parameter->convertTo0to1 (static_cast<float> (clamped)));
}

const juce::String AdvancedVSTiAudioProcessor::getProgramName (int index)
{
    const auto names = presetChoicesForFlavor();
    if (juce::isPositiveAndBelow (index, names.size()))
        return names[index];
    return {};
}

void AdvancedVSTiAudioProcessor::refreshSampleBank()
{
    const auto banks = sampleBankChoices();
    const auto targetBank = juce::jlimit (0, juce::jmax (0, banks.size() - 1), renderParams.sampleBank);
    if (loadedSampleBank == targetBank && loadedSample.getNumSamples() > 1)
        return;

    buildGeneratedSampleBank (targetBank);
    loadedSampleBank = targetBank;
}

void AdvancedVSTiAudioProcessor::buildGeneratedSampleBank (int bankIndex)
{
    const auto sampleRateF = static_cast<float> (currentSampleRate);
    if (sampleRateF <= 1000.0f)
    {
        loadedSample.setSize (1, 1);
        loadedSample.clear();
        return;
    }

    float lengthSec = 1.8f;
    switch (bankIndex)
    {
        case 1: lengthSec = 2.6f; break;
        case 2: lengthSec = 1.4f; break;
        case 3: lengthSec = 1.1f; break;
        case 4: lengthSec = 1.2f; break;
        case 5: lengthSec = 2.8f; break;
        default: break;
    }

    const auto numSamples = juce::jmax (256, juce::roundToInt (lengthSec * sampleRateF));
    loadedSample.setSize (1, numSamples);
    loadedSample.clear();

    auto* dst = loadedSample.getWritePointer (0);
    float phaseA = 0.0f;
    float phaseB = 0.0f;
    float phaseC = 0.0f;
    float phaseD = 0.0f;
    float stateA = 0.0f;

    for (int i = 0; i < numSamples; ++i)
    {
        const auto t = static_cast<float> (i) / sampleRateF;
        const auto noise = random.nextFloat() * 2.0f - 1.0f;
        float sample = 0.0f;

        switch (bankIndex)
        {
            case 1:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (523.2511f / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + (130.8128f / sampleRateF), 1.0f);
                phaseD = std::fmod (phaseD + (0.33f / sampleRateF), 1.0f);
                const auto swell = juce::jlimit (0.0f, 1.0f, t / 0.18f);
                const auto env = swell * std::exp (-t * 0.55f);
                const auto saw = (2.0f * phaseA) - 1.0f;
                const auto tri = 1.0f - (4.0f * std::abs (phaseB - 0.5f));
                sample = (saw * 0.46f) + (tri * 0.2f) + (std::sin (twoPi * phaseC) * 0.24f) + (noise * 0.01f);
                sample = smoothTowards (sample, 0.04f, stateA);
                sample += std::sin (twoPi * phaseD) * 0.07f * env;
                sample *= env;
                break;
            }
            case 2:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (527.0f / sampleRateF), 1.0f);
                const auto env = std::exp (-t * 5.4f);
                const auto click = std::exp (-t * 55.0f);
                sample = ((std::sin (twoPi * phaseA) * 0.7f) + (std::sin (twoPi * phaseB) * 0.28f)) * env;
                sample += noise * 0.17f * click;
                sample = softSaturate (sample * 1.22f);
                break;
            }
            case 3:
            {
                phaseA = std::fmod (phaseA + (220.0f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (660.0f / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + (1210.0f / sampleRateF), 1.0f);
                phaseD = std::fmod (phaseD + (330.0f / sampleRateF), 1.0f);
                const auto attack = juce::jlimit (0.0f, 1.0f, t / 0.03f);
                const auto env = attack * std::exp (-t * 1.9f);
                const auto vowel = (std::sin (twoPi * phaseA) * 0.4f) + (std::sin (twoPi * phaseB) * 0.22f) + (std::sin (twoPi * phaseC) * 0.13f);
                sample = smoothTowards (vowel + (noise * 0.05f), 0.09f, stateA);
                sample += std::sin (twoPi * phaseD) * 0.08f * std::exp (-std::pow ((t - 0.24f) * 4.2f, 2.0f));
                sample *= env;
                break;
            }
            case 4:
            {
                phaseA = std::fmod (phaseA + (65.4064f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (130.8128f / sampleRateF), 1.0f);
                const auto attack = juce::jlimit (0.0f, 1.0f, t * 28.0f);
                const auto env = attack * std::exp (-t * 3.4f);
                const auto square = phaseB < 0.5f ? 1.0f : -1.0f;
                sample = ((std::sin (twoPi * phaseA) * 0.78f) + (square * 0.14f)) * env;
                sample += noise * 0.04f * std::exp (-t * 44.0f);
                sample = softSaturate (sample * 1.35f);
                break;
            }
            case 5:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + ((261.6256f * 2.76f) / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + ((261.6256f * 4.17f) / sampleRateF), 1.0f);
                const auto env = std::exp (-t * 1.55f);
                const auto mod = std::sin (twoPi * phaseB) * (1.8f * std::exp (-t * 2.4f));
                const auto bell = std::sin ((twoPi * phaseA) + mod);
                const auto sparkle = std::sin (twoPi * phaseC) * std::exp (-t * 3.0f);
                sample = (bell * env) + (sparkle * 0.32f) + (noise * 0.012f * std::exp (-t * 7.0f));
                break;
            }
            case 0:
            default:
            {
                phaseA = std::fmod (phaseA + (261.6256f / sampleRateF), 1.0f);
                phaseB = std::fmod (phaseB + (523.2511f / sampleRateF), 1.0f);
                phaseC = std::fmod (phaseC + (784.8768f / sampleRateF), 1.0f);
                const auto env = std::exp (-t * 2.9f);
                const auto hammer = std::exp (-t * 38.0f);
                sample = ((std::sin (twoPi * phaseA) * 0.72f) + (std::sin (twoPi * phaseB) * 0.22f) + (std::sin (twoPi * phaseC) * 0.08f)) * env;
                sample += noise * (0.026f * env + 0.11f * hammer);
                sample = smoothTowards (sample, 0.08f, stateA);
                sample += std::sin (twoPi * std::fmod (phaseA * 0.5f, 1.0f)) * 0.05f * env;
                break;
            }
        }

        dst[i] = sample;
    }

    float peak = 0.0f;
    for (int i = 0; i < numSamples; ++i)
        peak = juce::jmax (peak, std::abs (dst[i]));

    if (peak > 0.0001f)
        juce::FloatVectorOperations::multiply (dst, 0.92f / peak, numSamples);
}

int AdvancedVSTiAudioProcessor::activeVoiceLimit() const noexcept
{
    if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
        return juce::jlimit (1, voiceLimitForFlavor(), renderParams.polyphony);

    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        return renderParams.monoEnabled ? 1 : voiceLimitForFlavor();

    return voiceLimitForFlavor();
}

void AdvancedVSTiAudioProcessor::handleMidiMessage (const juce::MidiMessage& msg)
{
    if (msg.isNoteOn())
    {
        const auto polyphonyLimit = activeVoiceLimit();
        int externalPadIndex = -1;
        if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
        {
            externalPadIndex = externalPadIndexForMidi (msg.getNoteNumber());
            if (externalPadIndex < 0)
                return;
        }

        if constexpr (isSynthDrumFlavor())
        {
            const auto triggeredKind = drumKindForMidi (msg.getNoteNumber());
            if (isHatKind (triggeredKind))
            {
                for (auto& activeVoice : voices)
                {
                    if (! activeVoice.active)
                        continue;

                    if (isHatKind (drumKindForMidi (activeVoice.midiNote)))
                    {
                        activeVoice.active = false;
                        activeVoice.ampEnv.reset();
                        activeVoice.filterEnv.reset();
                    }
                }
            }
        }

        int voiceIndex = 0;
        bool foundVoice = false;
        float oldestVoiceAge = -1.0f;

        if constexpr (isMonophonicFlavor())
        {
            heldNotes.clear();
            for (auto& activeVoice : voices)
            {
                if (! activeVoice.active)
                    continue;

                activeVoice.active = false;
                activeVoice.ampEnv.reset();
                activeVoice.filterEnv.reset();
            }
        }

        for (int i = 0; i < polyphonyLimit; ++i)
        {
            auto& candidate = voices[static_cast<size_t> (i)];
            if (! candidate.active)
            {
                voiceIndex = i;
                foundVoice = true;
                break;
            }

            if (candidate.noteAge > oldestVoiceAge)
            {
                oldestVoiceAge = candidate.noteAge;
                voiceIndex = i;
            }
        }

        auto& voice = voices[static_cast<size_t> (voiceIndex)];
        if (! foundVoice && voice.active)
        {
            voice.ampEnv.reset();
            voice.filterEnv.reset();
        }
        voice.active = true;
        voice.midiNote = msg.getNoteNumber();
        voice.velocity = msg.getFloatVelocity();
        voice.phase = 0.0f;
        voice.osc2Phase = 0.0f;
        voice.subPhase = 0.0f;
        voice.fmPhase = 0.0f;
        voice.syncPhase = 0.0f;
        voice.samplePos = 0.0f;
        voice.noteAge = 0.0f;
        voice.auxPhase = 0.0f;
        voice.toneState = 0.0f;
        voice.colourState = 0.0f;
        voice.externalPadIndex = externalPadIndex;
        voice.externalSamplePosition = 0.0;
        voice.externalSample = (externalPadIndex >= 0 && juce::isPositiveAndBelow (externalPadIndex, vecPadCount))
                                   ? std::atomic_load (&externalPadSamples[static_cast<size_t> (externalPadIndex)])
                                   : std::shared_ptr<const ExternalSampleData> {};
        voice.unisonPhases.fill (0.0f);
        voice.unisonSyncPhases.fill (0.0f);
        voice.unisonSamplePositions.fill (0.0f);
        voice.ampEnv.noteOn();
        voice.filterEnv.noteOn();

        if (renderParams.lfo1EnvMode)
            lfo1Phase = 0.0f;
        if (renderParams.lfo2EnvMode)
            lfo2Phase = 0.0f;
        if (renderParams.lfo3EnvMode)
            lfo3Phase = 0.0f;

        if constexpr (! isDrumFlavor())
            heldNotes.addIfNotAlreadyThere (voice.midiNote);
        return;
    }

    if (msg.isNoteOff())
    {
        if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
        {
            const auto note = msg.getNoteNumber();
            for (auto& voice : voices)
            {
                if (! voice.active || voice.midiNote != note || ! juce::isPositiveAndBelow (voice.externalPadIndex, vecPadCount))
                    continue;

                const auto padIndex = static_cast<size_t> (voice.externalPadIndex);
                const auto sustainTime = juce::jmax (0.0f, renderParams.externalPadSustainTimes[padIndex]);
                const auto releaseTime = juce::jmax (0.0f, renderParams.externalPadReleaseTimes[padIndex]);
                if (releaseTime <= 0.0f)
                {
                    voice.active = false;
                }
                else
                {
                    voice.noteAge = juce::jmax (voice.noteAge, sustainTime);
                }
            }
            return;
        }

        if constexpr (isDrumFlavor())
            return;

        const auto note = msg.getNoteNumber();
        const bool keepLatchedForArp = arpHoldEnabled.load() && renderParams.arpEnabled;
        if (keepLatchedForArp)
            return;

        heldNotes.removeAllInstancesOf (note);
        for (auto& voice : voices)
        {
            if (voice.active && voice.midiNote == note)
            {
                voice.ampEnv.noteOff();
                voice.filterEnv.noteOff();
            }
        }
    }
}

float AdvancedVSTiAudioProcessor::lfoValue (int shape, float phase) const
{
    const auto wrapped = phase - std::floor (phase);
    if (shape == 1)
        return 1.0f - 4.0f * std::abs (wrapped - 0.5f);
    if (shape == 2)
        return 2.0f * wrapped - 1.0f;
    if (shape == 3)
    {
        const auto hashed = std::sin ((wrapped + 1.0f) * 12345.6789f) * 43758.5453f;
        return 2.0f * (hashed - std::floor (hashed)) - 1.0f;
    }
    if (shape == 4)
        return wrapped < 0.5f ? 1.0f : -1.0f;
    return std::sin (twoPi * wrapped);
}

float AdvancedVSTiAudioProcessor::fmOperator (VoiceState& voice, float baseFreq, float amount)
{
    const auto increment = baseFreq / static_cast<float> (currentSampleRate);
    voice.fmPhase = std::fmod (voice.fmPhase + increment, 1.0f);
    return std::sin (twoPi * voice.fmPhase) * amount;
}

float AdvancedVSTiAudioProcessor::oscSample (VoiceState& voice, float baseFreq, OscType type, float syncAmount, float pulseWidth)
{
    return oscSampleForState (voice.phase, voice.syncPhase, voice.samplePos, baseFreq, type, syncAmount, pulseWidth);
}

float AdvancedVSTiAudioProcessor::oscSampleForState (float& phase, float& syncPhase, float& samplePos,
                                                     float baseFreq, OscType type, float syncAmount, float pulseWidth)
{
    auto increment = baseFreq / static_cast<float> (currentSampleRate);
    const auto clampedPulseWidth = juce::jlimit (0.05f, 0.95f, pulseWidth);

    syncPhase = std::fmod (syncPhase + increment * (1.0f + syncAmount), 1.0f);
    if (syncPhase < increment)
        phase = 0.0f;

    phase = std::fmod (phase + increment, 1.0f);

    switch (type)
    {
        case OscType::saw: return 2.0f * phase - 1.0f;
        case OscType::square: return phase < clampedPulseWidth ? 1.0f : -1.0f;
        case OscType::noise: return random.nextFloat() * 2.0f - 1.0f;
        case OscType::hypersaw: return hypersawSample (phase, clampedPulseWidth);
        case OscType::wavetableFormant: return wavetableOscSample (phase, baseFreq, clampedPulseWidth, 0);
        case OscType::wavetableComplex: return wavetableOscSample (phase, baseFreq, clampedPulseWidth, 1);
        case OscType::wavetableMetal: return wavetableOscSample (phase, baseFreq, clampedPulseWidth, 2);
        case OscType::wavetableVocal: return wavetableOscSample (phase, baseFreq, clampedPulseWidth, 3);
        case OscType::sample:
        {
            if (loadedSample.getNumSamples() <= 1)
                refreshSampleBank();
            if (loadedSample.getNumSamples() <= 1)
                return 0.0f;
            const auto totalSamples = loadedSample.getNumSamples();
            const auto loopStart = juce::jlimit (0, totalSamples - 1, juce::roundToInt (renderParams.sampleStart * static_cast<float> (totalSamples - 1)));
            const auto loopEnd = juce::jlimit (loopStart + 1, totalSamples, juce::roundToInt (renderParams.sampleEnd * static_cast<float> (totalSamples)));
            if (samplePos < static_cast<float> (loopStart) || samplePos >= static_cast<float> (loopEnd))
                samplePos = static_cast<float> (loopStart);
            const auto idx = juce::jlimit (loopStart, loopEnd - 1, static_cast<int> (samplePos));
            const auto sample = loadedSample.getSample (0, idx);
            const auto playbackRatio = juce::jlimit (0.125f, 8.0f, baseFreq / 261.6256f);
            samplePos += playbackRatio;
            if (samplePos >= static_cast<float> (loopEnd))
                samplePos = static_cast<float> (loopStart);
            return sample;
        }
        case OscType::sine:
        default:
            return std::sin (twoPi * phase);
    }
}

float AdvancedVSTiAudioProcessor::basicOscSample (float& phase, float frequency, OscType type, float pulseWidth)
{
    const auto increment = frequency / static_cast<float> (currentSampleRate);
    const auto clampedPulseWidth = juce::jlimit (0.05f, 0.95f, pulseWidth);
    phase = std::fmod (phase + increment, 1.0f);

    switch (type)
    {
        case OscType::saw: return (2.0f * phase) - 1.0f;
        case OscType::square: return phase < clampedPulseWidth ? 1.0f : -1.0f;
        case OscType::noise: return random.nextFloat() * 2.0f - 1.0f;
        case OscType::hypersaw: return hypersawSample (phase, clampedPulseWidth);
        case OscType::wavetableFormant: return wavetableOscSample (phase, frequency, clampedPulseWidth, 0);
        case OscType::wavetableComplex: return wavetableOscSample (phase, frequency, clampedPulseWidth, 1);
        case OscType::wavetableMetal: return wavetableOscSample (phase, frequency, clampedPulseWidth, 2);
        case OscType::wavetableVocal: return wavetableOscSample (phase, frequency, clampedPulseWidth, 3);
        case OscType::sample:
        case OscType::sine:
        default:
            return std::sin (twoPi * phase);
    }
}

void AdvancedVSTiAudioProcessor::applyPendingUiActions (juce::MidiBuffer& midiMessages, int blockSamples)
{
    auto releaseAllNotes = [this] (bool hardKill)
    {
        heldNotes.clear();
        arpStep = 0;
        keyboardState.reset();

        for (auto& voice : voices)
        {
            if (! voice.active)
                continue;

            if (hardKill)
            {
                voice.active = false;
                voice.midiNote = -1;
                voice.velocity = 0.0f;
                voice.ampEnv.reset();
                voice.filterEnv.reset();
            }
            else
            {
                voice.ampEnv.noteOff();
                voice.filterEnv.noteOff();
            }
        }
    };

    if (pendingPanicAllNotes.exchange (false))
    {
        pendingReleaseHeldNotes.store (false);
        activeAuditionNote = -1;
        auditionSamplesRemaining = 0;
        releaseAllNotes (true);
    }
    else if (pendingReleaseHeldNotes.exchange (false))
    {
        activeAuditionNote = -1;
        auditionSamplesRemaining = 0;
        releaseAllNotes (false);
    }

    if (activeAuditionNote >= 0)
    {
        auditionSamplesRemaining -= blockSamples;
        if (auditionSamplesRemaining <= 0)
        {
            midiMessages.addEvent (juce::MidiMessage::noteOff (1, activeAuditionNote), 0);
            activeAuditionNote = -1;
            auditionSamplesRemaining = 0;
        }
    }

    const auto requestedNote = pendingAuditionNote.exchange (-1);
    if (requestedNote >= 0)
    {
        if (activeAuditionNote >= 0)
            midiMessages.addEvent (juce::MidiMessage::noteOff (1, activeAuditionNote), 0);

        const auto velocity = static_cast<juce::uint8> (juce::jlimit (1, 127, pendingAuditionVelocity.load()));
        midiMessages.addEvent (juce::MidiMessage::noteOn (1, requestedNote, velocity), 0);
        activeAuditionNote = requestedNote;
        auditionSamplesRemaining = juce::jmax (1,
                                               juce::roundToInt (currentSampleRate
                                                                 * static_cast<double> (juce::jmax (80, pendingAuditionDurationMs.load()))
                                                                 / 1000.0));
    }
}

float AdvancedVSTiAudioProcessor::hypersawSample (float phase, float shape) const
{
    constexpr std::array<float, 5> offsets { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
    constexpr std::array<float, 5> weights { 0.18f, 0.23f, 0.28f, 0.23f, 0.18f };

    const auto spread = juce::jmap (juce::jlimit (0.05f, 0.95f, shape), 0.004f, 0.12f);
    float sample = 0.0f;

    for (size_t index = 0; index < offsets.size(); ++index)
    {
        const auto detunedPhase = std::fmod (phase + offsets[index] * spread + 1.0f, 1.0f);
        sample += ((2.0f * detunedPhase) - 1.0f) * weights[index];
    }

    return juce::jlimit (-1.0f, 1.0f, sample);
}

float AdvancedVSTiAudioProcessor::wavetableOscSample (float& phase, float frequency, float shape, int variant)
{
    const auto position = juce::jmap (juce::jlimit (0.05f, 0.95f, shape), 0.05f, 0.95f, 0.0f, 1.0f);
    const auto& frames = virusWavetableFramesForVariant (variant);
    const auto framePosition = position * static_cast<float> (frames.size() - 1);
    const auto frameA = juce::jlimit (0, static_cast<int> (frames.size()) - 1, static_cast<int> (std::floor (framePosition)));
    const auto frameB = juce::jlimit (0, static_cast<int> (frames.size()) - 1, frameA + 1);
    const auto blend = framePosition - static_cast<float> (frameA);

    const auto sampleA = sampleVirusWavetableFrame (phase,
                                                    frequency,
                                                    currentSampleRate,
                                                    frames[static_cast<size_t> (frameA)]);
    const auto sampleB = sampleVirusWavetableFrame (phase,
                                                    frequency,
                                                    currentSampleRate,
                                                    frames[static_cast<size_t> (frameB)]);

    return juce::jlimit (-1.0f, 1.0f, juce::jmap (blend, sampleA, sampleB));
}

float AdvancedVSTiAudioProcessor::renderDrumVoiceSample (VoiceState& voice)
{
    if (! voice.active)
        return 0.0f;

    const auto age = voice.noteAge;
    const auto dt = 1.0f / static_cast<float> (currentSampleRate);
    const auto noise = random.nextFloat() * 2.0f - 1.0f;
    const auto kind = drumKindForMidi (voice.midiNote);
    const auto voiceIndex = drumVoiceIndex (kind);
    const auto voiceTune = renderParams.detune + renderParams.drumVoiceTunes[voiceIndex];
    const auto tuneRatio = std::pow (2.0f, voiceTune / 12.0f);
    const auto accent = juce::jlimit (0.0f, 1.0f, renderParams.fmAmount / 200.0f);
    const auto punch = accent;
    const auto attack = juce::jlimit (0.0f, 1.0f, renderParams.syncAmount);
    const auto kickAttack = juce::jlimit (0.0f, 1.0f, 0.35f * attack + 0.65f * renderParams.drumKickAttack);
    const auto snareTone = juce::jlimit (0.0f, 1.0f, renderParams.drumSnareTone);
    const auto snareSnappy = juce::jlimit (0.0f, 1.0f, renderParams.drumSnareSnappy);
    const auto decayMacro = std::sqrt (juce::jlimit (0.0f, 1.0f, renderParams.gateLength / 2.4f));
    const auto decayScale = juce::jmap (decayMacro, 0.68f, buildFlavor() == InstrumentFlavor::drum808 ? 2.25f : 1.95f);
    const auto voiceDecayScale = 0.4f + renderParams.drumVoiceDecays[voiceIndex] * 1.2f;
    const auto brightness = normalizedLogValue (renderParams.cutoff, 120.0f, 20000.0f);
    const auto noiseMacro = juce::jlimit (0.0f, 1.0f, renderParams.filterEnvAmount);
    const auto ring = juce::jlimit (0.0f, 1.0f, (renderParams.resonance - 0.1f) / 1.1f);
    const auto pitchScale = tuneRatio * juce::jmap (brightness, 0.94f, 1.08f);

    float output = 0.0f;
    float duration = 0.4f;
    auto scaledHz = [pitchScale] (float hz) -> float
    {
        return hz * pitchScale;
    };
    auto advanceSine = [dt] (float& phase, float hz) -> float
    {
        phase = std::fmod (phase + hz * dt, 1.0f);
        return std::sin (twoPi * phase);
    };
    auto advanceSquare = [dt] (float& phase, float hz) -> float
    {
        phase = std::fmod (phase + hz * dt, 1.0f);
        return phase < 0.5f ? 1.0f : -1.0f;
    };

    if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        switch (kind)
        {
            case DrumVoiceKind::kick:
            {
                duration = 1.45f;
                const auto pitch = scaledHz (34.0f + (82.0f + (punch * 26.0f)) * std::exp (-age * (7.5f + kickAttack * 5.0f)));
                voice.auxPhase = std::fmod (voice.auxPhase + pitch * dt, 1.0f);
                const auto body = std::sin (twoPi * voice.auxPhase) * std::exp (-age * (2.2f + ring * 0.6f));
                const auto sub = std::sin (twoPi * std::fmod (voice.auxPhase * 0.5f, 1.0f)) * std::exp (-age * 1.8f);
                const auto beater = noise * (0.03f + punch * 0.06f + kickAttack * 0.06f) * std::exp (-age * 95.0f);
                output = (body * 0.92f) + (sub * 0.42f) + beater;
                break;
            }
            case DrumVoiceKind::snare:
            {
                duration = 0.62f;
                const auto toneScale = juce::jmap (snareTone, 0.84f, 1.28f);
                const auto body = ((advanceSine (voice.phase, scaledHz (182.0f * toneScale)) * 0.6f)
                                   + (advanceSine (voice.fmPhase, scaledHz (331.0f * toneScale)) * 0.38f)) * std::exp (-age * (8.4f - snareTone * 1.8f));
                const auto rattle = noise * std::exp (-age * (14.0f - brightness * 2.0f - snareSnappy * 3.0f));
                output = (body * juce::jmap (snareTone, 0.66f, 0.52f)) + (rattle * juce::jmap (snareSnappy, 0.62f, 0.96f));
                break;
            }
            case DrumVoiceKind::rim:
            {
                duration = 0.12f;
                const auto wood = (advanceSine (voice.phase, scaledHz (1710.0f)) * 0.54f) + (advanceSine (voice.fmPhase, scaledHz (2440.0f)) * 0.24f);
                const auto click = noise * std::exp (-age * 78.0f);
                output = (wood * std::exp (-age * 26.0f)) + (click * 0.12f);
                break;
            }
            case DrumVoiceKind::clap:
            {
                duration = 0.5f;
                const auto pulseA = std::exp (-std::pow ((age - 0.00f) * 38.0f, 2.0f));
                const auto pulseB = std::exp (-std::pow ((age - 0.034f) * 34.0f, 2.0f));
                const auto pulseC = std::exp (-std::pow ((age - 0.072f) * 30.0f, 2.0f));
                const auto tail = std::exp (-age * 7.6f);
                output = noise * ((0.82f * pulseA) + (0.78f * pulseB) + (0.66f * pulseC) + (0.4f * tail));
                break;
            }
            case DrumVoiceKind::closedHat:
            case DrumVoiceKind::openHat:
            {
                duration = kind == DrumVoiceKind::closedHat ? 0.16f : 0.72f;
                const auto metallic =
                    advanceSine (voice.phase, scaledHz (4180.0f))
                    + advanceSine (voice.fmPhase, scaledHz (6450.0f))
                    + advanceSine (voice.auxPhase, scaledHz (8630.0f))
                    + advanceSine (voice.syncPhase, scaledHz (10940.0f));
                const auto decayWeight = juce::jmap (voiceDecayScale, 1.18f, 0.74f);
                const auto env = std::exp (-age * (kind == DrumVoiceKind::closedHat ? 34.0f : 10.5f) / decayWeight);
                const auto airy = noise * std::exp (-age * (kind == DrumVoiceKind::closedHat ? 42.0f : 12.5f) / decayWeight);
                output = (metallic * 0.13f + airy * 0.78f) * env;
                break;
            }
            case DrumVoiceKind::lowTom:
            case DrumVoiceKind::midTom:
            case DrumVoiceKind::highTom:
            {
                const auto basePitch = kind == DrumVoiceKind::lowTom ? 98.0f
                                       : kind == DrumVoiceKind::midTom ? 146.0f
                                                                       : 198.0f;
                duration = kind == DrumVoiceKind::lowTom ? 1.0f
                           : kind == DrumVoiceKind::midTom ? 0.86f
                                                           : 0.7f;
                const auto pitch = scaledHz (juce::jlimit (72.0f, 250.0f, basePitch + (voice.midiNote - 45) * 1.6f));
                const auto sweep = 1.0f + 0.42f * std::exp (-age * 5.4f);
                const auto fundamental = advanceSine (voice.auxPhase, pitch * sweep);
                const auto overtone = advanceSine (voice.syncPhase, (pitch * 1.49f) * (0.95f + std::exp (-age * 4.6f) * 0.1f));
                output = (fundamental * 0.94f * std::exp (-age * 3.4f)) + (overtone * 0.18f * std::exp (-age * 4.1f));
                break;
            }
            case DrumVoiceKind::crash:
            {
                duration = 1.28f;
                const auto cluster =
                    advanceSine (voice.phase, scaledHz (2890.0f))
                    + advanceSine (voice.fmPhase, scaledHz (4310.0f))
                    + advanceSine (voice.auxPhase, scaledHz (6120.0f))
                    + advanceSine (voice.syncPhase, scaledHz (9150.0f));
                const auto wash = noise * (0.86f + brightness * 0.08f);
                output = (cluster * 0.11f + wash * 0.82f) * std::exp (-age * 2.9f);
                output += noise * 0.08f * std::exp (-age * 10.0f);
                break;
            }
            case DrumVoiceKind::ride:
            {
                duration = 1.85f;
                const auto bell = advanceSine (voice.phase, scaledHz (612.0f));
                const auto metallic =
                    advanceSine (voice.fmPhase, scaledHz (3020.0f))
                    + advanceSine (voice.auxPhase, scaledHz (4680.0f))
                    + advanceSine (voice.syncPhase, scaledHz (7010.0f));
                output = (bell * 0.34f * std::exp (-age * 2.1f)) + ((metallic * 0.16f) + (noise * 0.22f)) * std::exp (-age * 2.7f);
                break;
            }
            case DrumVoiceKind::cowbell:
            {
                duration = 0.42f;
                const auto bellA = advanceSquare (voice.phase, scaledHz (541.0f));
                const auto bellB = advanceSquare (voice.fmPhase, scaledHz (811.0f));
                const auto env = std::exp (-age * 8.8f);
                output = ((bellA * 0.46f) + (bellB * 0.34f)) * env;
                break;
            }
            case DrumVoiceKind::clave:
            {
                duration = 0.14f;
                const auto body = advanceSine (voice.phase, scaledHz (2470.0f)) * std::exp (-age * 28.0f);
                const auto click = noise * std::exp (-age * 96.0f);
                output = body + (click * 0.05f);
                break;
            }
            case DrumVoiceKind::maraca:
            {
                duration = 0.18f;
                const auto shaker = noise * std::exp (-age * 20.0f);
                const auto grain = advanceSine (voice.phase, scaledHz (5200.0f)) * 0.08f * std::exp (-age * 26.0f);
                output = shaker * 0.82f + grain;
                break;
            }
            case DrumVoiceKind::perc:
            default:
            {
                duration = 0.38f;
                const auto squareA = advanceSquare (voice.phase, scaledHz (540.0f));
                const auto squareB = advanceSquare (voice.fmPhase, scaledHz (846.0f));
                output = (squareA * 0.42f + squareB * 0.34f) * std::exp (-age * 9.4f);
                output += noise * 0.08f * std::exp (-age * 20.0f);
                break;
            }
        }
    }
    else
    {
        switch (kind)
        {
            case DrumVoiceKind::kick:
            {
                duration = 1.0f;
                const auto pitch = scaledHz (46.0f + (126.0f + (punch * 40.0f)) * std::exp (-age * (10.5f + kickAttack * 4.8f)));
                voice.auxPhase = std::fmod (voice.auxPhase + pitch * dt, 1.0f);
                const auto tone = std::sin (twoPi * voice.auxPhase) * std::exp (-age * (4.7f + ring * 1.3f));
                const auto thump = std::sin (twoPi * std::fmod (voice.auxPhase * 0.5f, 1.0f)) * 0.22f * std::exp (-age * 3.6f);
                const auto click = noise * (0.13f + punch * 0.11f + kickAttack * 0.07f) * std::exp (-age * 85.0f);
                output = tone + thump + click;
                break;
            }
            case DrumVoiceKind::snare:
            {
                duration = 0.48f;
                const auto toneScale = juce::jmap (snareTone, 0.82f, 1.22f);
                const auto body = ((advanceSine (voice.phase, scaledHz (186.0f * toneScale)) * 0.56f)
                                   + (advanceSine (voice.fmPhase, scaledHz (329.0f * toneScale)) * 0.31f)) * std::exp (-age * (10.6f - snareTone * 2.2f));
                const auto sizzle = noise * std::exp (-age * (17.0f - brightness * 4.0f - snareSnappy * 3.5f));
                output = (body * juce::jmap (snareTone, 0.68f, 0.52f)) + (sizzle * juce::jmap (snareSnappy, 0.66f, 1.02f));
                break;
            }
            case DrumVoiceKind::rim:
            {
                duration = 0.11f;
                const auto crack = (advanceSine (voice.phase, scaledHz (1825.0f)) * 0.46f) + (advanceSine (voice.fmPhase, scaledHz (2510.0f)) * 0.26f);
                const auto click = noise * std::exp (-age * 92.0f);
                output = (crack * std::exp (-age * 30.0f)) + (click * 0.08f);
                break;
            }
            case DrumVoiceKind::clap:
            {
                duration = 0.36f;
                const auto pulseA = std::exp (-std::pow ((age - 0.00f) * 44.0f, 2.0f));
                const auto pulseB = std::exp (-std::pow ((age - 0.032f) * 42.0f, 2.0f));
                const auto pulseC = std::exp (-std::pow ((age - 0.068f) * 36.0f, 2.0f));
                const auto tail = std::exp (-age * 9.4f);
                output = noise * ((0.86f * pulseA) + (0.74f * pulseB) + (0.58f * pulseC) + (0.32f * tail));
                break;
            }
            case DrumVoiceKind::closedHat:
            case DrumVoiceKind::openHat:
            {
                duration = kind == DrumVoiceKind::closedHat ? 0.13f : 0.56f;
                const auto metallic =
                    advanceSine (voice.phase, scaledHz (4620.0f))
                    + advanceSine (voice.fmPhase, scaledHz (6840.0f))
                    + advanceSine (voice.auxPhase, scaledHz (9470.0f))
                    + advanceSine (voice.syncPhase, scaledHz (12030.0f));
                const auto body = metallic * 0.1f;
                const auto air = noise * (0.74f + brightness * 0.14f);
                const auto decayWeight = juce::jmap (voiceDecayScale, 1.18f, 0.74f);
                const auto env = std::exp (-age * (kind == DrumVoiceKind::closedHat ? 44.0f : 12.8f) / decayWeight);
                output = (body + air) * env;
                break;
            }
            case DrumVoiceKind::lowTom:
            case DrumVoiceKind::midTom:
            case DrumVoiceKind::highTom:
            {
                const auto basePitch = kind == DrumVoiceKind::lowTom ? 112.0f
                                       : kind == DrumVoiceKind::midTom ? 162.0f
                                                                       : 222.0f;
                duration = kind == DrumVoiceKind::lowTom ? 0.82f
                           : kind == DrumVoiceKind::midTom ? 0.66f
                                                           : 0.5f;
                const auto pitch = scaledHz (juce::jlimit (90.0f, 280.0f, basePitch + (voice.midiNote - 45) * 1.8f));
                const auto sweep = 0.84f + 0.31f * std::exp (-age * 6.2f);
                const auto ringTone = advanceSine (voice.auxPhase, pitch * sweep) * std::exp (-age * 5.1f);
                const auto overtone = advanceSine (voice.syncPhase, pitch * 1.52f) * 0.18f * std::exp (-age * 7.2f);
                output = ringTone + overtone + noise * 0.05f * std::exp (-age * 24.0f);
                break;
            }
            case DrumVoiceKind::crash:
            {
                duration = 1.14f;
                const auto metallic =
                    advanceSine (voice.phase, scaledHz (3420.0f))
                    + advanceSine (voice.fmPhase, scaledHz (5180.0f))
                    + advanceSine (voice.auxPhase, scaledHz (7760.0f))
                    + advanceSine (voice.syncPhase, scaledHz (10520.0f));
                const auto air = noise * (0.82f + brightness * 0.12f);
                output = (metallic * 0.12f + air * 0.78f) * std::exp (-age * 3.8f);
                break;
            }
            case DrumVoiceKind::ride:
            {
                duration = 1.72f;
                const auto bell = advanceSine (voice.phase, scaledHz (680.0f));
                const auto metallic =
                    advanceSine (voice.fmPhase, scaledHz (3510.0f))
                    + advanceSine (voice.auxPhase, scaledHz (5630.0f))
                    + advanceSine (voice.syncPhase, scaledHz (8240.0f));
                output = (bell * 0.32f * std::exp (-age * 2.4f)) + ((metallic * 0.14f) + (noise * 0.18f)) * std::exp (-age * 3.0f);
                break;
            }
            case DrumVoiceKind::cowbell:
            {
                duration = 0.34f;
                const auto bellA = advanceSquare (voice.phase, scaledHz (587.0f));
                const auto bellB = advanceSquare (voice.fmPhase, scaledHz (845.0f));
                const auto env = std::exp (-age * 10.5f);
                output = ((bellA * 0.4f) + (bellB * 0.28f)) * env;
                break;
            }
            case DrumVoiceKind::clave:
            {
                duration = 0.12f;
                const auto body = advanceSine (voice.phase, scaledHz (2860.0f)) * std::exp (-age * 36.0f);
                const auto click = noise * std::exp (-age * 120.0f);
                output = body + (click * 0.04f);
                break;
            }
            case DrumVoiceKind::maraca:
            {
                duration = 0.16f;
                const auto shaker = noise * std::exp (-age * 24.0f);
                const auto grit = advanceSine (voice.phase, scaledHz (6100.0f)) * 0.05f * std::exp (-age * 28.0f);
                output = shaker * 0.88f + grit;
                break;
            }
            case DrumVoiceKind::perc:
            default:
            {
                duration = 0.28f;
                const auto metallic = (advanceSine (voice.phase, scaledHz (460.0f)) * 0.52f) + (advanceSine (voice.fmPhase, scaledHz (744.0f)) * 0.31f);
                const auto chop = advanceSquare (voice.syncPhase, scaledHz (1220.0f));
                output = (metallic + chop * 0.16f) * std::exp (-age * 13.0f);
                output += noise * 0.11f * std::exp (-age * 26.0f);
                break;
            }
        }
    }

    const auto transientDrive = kind == DrumVoiceKind::kick ? kickAttack : attack;
    duration *= decayScale * voiceDecayScale;
    output *= std::exp (-age * juce::jmap (decayMacro, 10.5f, 1.15f) / voiceDecayScale);
    output += noise * drumTransientWeight (kind) * (transientDrive * 0.45f + punch * 0.12f) * std::exp (-age * 120.0f);
    output += noise * drumNoiseWeight (kind) * noiseMacro * std::exp (-age * juce::jmap (brightness, 30.0f, 12.0f));

    const auto toneCoeff = juce::jmap (brightness, 0.035f, 0.22f);
    const auto body = smoothTowards (output, toneCoeff, voice.colourState);
    const auto edge = output - body;
    output = (body * juce::jmap (brightness, 1.2f, 0.8f)) + (edge * juce::jmap (brightness, 0.3f, 1.7f));
    output = softSaturate (output * (1.08f + punch * 0.26f + noiseMacro * 0.08f)) * (buildFlavor() == InstrumentFlavor::drum808 ? 0.94f : 0.92f);

    voice.noteAge += dt;
    if (voice.noteAge >= duration)
        voice.active = false;

    return output * voice.velocity * renderParams.drumVoiceLevels[voiceIndex] * renderParams.drumMasterLevel;
}

float AdvancedVSTiAudioProcessor::renderExternalPadVoiceSample (VoiceState& voice)
{
    if (! voice.active)
        return 0.0f;

    const auto sampleData = voice.externalSample;
    if (sampleData == nullptr || sampleData->audio.getNumSamples() <= 0 || voice.externalPadIndex < 0)
    {
        voice.active = false;
        return 0.0f;
    }

    const auto totalSamples = sampleData->audio.getNumSamples();
    if (voice.externalSamplePosition >= static_cast<double> (totalSamples))
    {
        voice.active = false;
        return 0.0f;
    }

    const auto indexA = juce::jlimit (0, totalSamples - 1, static_cast<int> (voice.externalSamplePosition));
    const auto indexB = juce::jmin (totalSamples - 1, indexA + 1);
    const auto alpha = static_cast<float> (voice.externalSamplePosition - static_cast<double> (indexA));
    const auto sampleA = sampleData->audio.getSample (0, indexA);
    const auto sampleB = sampleData->audio.getSample (0, indexB);
    const auto output = juce::jmap (alpha, sampleA, sampleB);
    const auto padIndex = juce::jlimit (0, vecPadCount - 1, voice.externalPadIndex);
    const auto sustainTime = juce::jmax (0.0f, renderParams.externalPadSustainTimes[static_cast<size_t> (padIndex)]);
    const auto releaseTime = juce::jmax (0.0f, renderParams.externalPadReleaseTimes[static_cast<size_t> (padIndex)]);
    auto envelope = 1.0f;

    if (releaseTime <= 0.0f)
    {
        if (voice.noteAge > sustainTime)
        {
            voice.active = false;
            return 0.0f;
        }
    }
    else if (voice.noteAge > sustainTime)
    {
        envelope = juce::jlimit (0.0f, 1.0f, 1.0f - ((voice.noteAge - sustainTime) / releaseTime));
        if (envelope <= 0.0f)
        {
            voice.active = false;
            return 0.0f;
        }
    }

    const auto increment = juce::jlimit (0.125, 8.0, sampleData->sampleRate / currentSampleRate);

    voice.externalSamplePosition += increment;
    voice.noteAge += 1.0f / static_cast<float> (currentSampleRate);
    if (voice.externalSamplePosition >= static_cast<double> (totalSamples))
        voice.active = false;
    if (releaseTime <= 0.0f)
    {
        if (voice.noteAge > sustainTime)
            voice.active = false;
    }
    else if (voice.noteAge > (sustainTime + releaseTime))
    {
        voice.active = false;
    }

    const auto padLevel = renderParams.externalPadLevels[static_cast<size_t> (padIndex)];
    return output * envelope * voice.velocity * padLevel * renderParams.drumMasterLevel;
}

float AdvancedVSTiAudioProcessor::renderVoiceSample (VoiceState& voice, SampleModulationSums& sampleModSums)
{
    if (! voice.active)
        return 0.0f;

    if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
        return renderExternalPadVoiceSample (voice);

    if constexpr (isSynthDrumFlavor())
        return renderDrumVoiceSample (voice);

    const auto& params = renderParams;

    const auto lfoPhaseFor = [&] (int index) -> float
    {
        switch (index)
        {
            case 1: return params.lfo2EnvMode ? std::fmod (params.lfo2Rate * voice.noteAge, 1.0f) : lfo2Phase;
            case 2: return params.lfo3EnvMode ? std::fmod (params.rhythmGateRate * voice.noteAge, 1.0f) : lfo3Phase;
            default: return params.lfo1EnvMode ? std::fmod (params.lfo1Rate * voice.noteAge, 1.0f) : lfo1Phase;
        }
    };

    const auto lfo1ValueNow = params.lfo1Enabled ? lfoValue (params.lfo1Shape, lfoPhaseFor (0)) : 0.0f;
    const auto lfo2ValueNow = params.lfo2Enabled ? lfoValue (params.lfo2Shape, lfoPhaseFor (1)) : 0.0f;
    const auto lfo3ValueNow = params.lfo3Enabled ? lfoValue (params.lfo3Shape, lfoPhaseFor (2)) : 0.0f;
    const auto ampEnv = shapedEnv (voice.ampEnv.getNextSample(), params.envCurve);
    const auto filtEnv = shapedEnv (voice.filterEnv.getNextSample(), params.envCurve);

    struct VoiceModulation
    {
        float osc1Pitch = 0.0f;
        float osc23Pitch = 0.0f;
        float pulseWidth = 0.0f;
        float shape = 0.0f;
        float fmAmount = 0.0f;
        float detune = 0.0f;
        float syncAmount = 0.0f;
        float gateLength = 0.0f;
        float filterGain = 0.0f;
        float ampLevel = 0.0f;
        float oscVolume = 0.0f;
        float subOscVolume = 0.0f;
        float noiseVolume = 0.0f;
        float osc2Mix = 0.0f;
        float ringModAmount = 0.0f;
    } voiceMod;

    auto applyMatrixDestination = [&] (int destination, float value)
    {
        switch (static_cast<MatrixDestination> (destination))
        {
            case MatrixDestination::osc1Pitch: voiceMod.osc1Pitch += value * 24.0f; break;
            case MatrixDestination::osc23Pitch: voiceMod.osc23Pitch += value * 24.0f; break;
            case MatrixDestination::pulseWidth: voiceMod.pulseWidth += value * 0.35f; break;
            case MatrixDestination::resonance: sampleModSums.resonance += value * 0.8f; break;
            case MatrixDestination::filterGain: voiceMod.filterGain += value * 0.9f; break;
            case MatrixDestination::cutoff1: sampleModSums.cutoff1 += value * 6000.0f; break;
            case MatrixDestination::cutoff2: sampleModSums.cutoff2 += value * 6000.0f; break;
            case MatrixDestination::shape: voiceMod.shape += value * 0.4f; break;
            case MatrixDestination::fmAmount: voiceMod.fmAmount += value * 800.0f; break;
            case MatrixDestination::panorama: sampleModSums.panorama += value * 0.9f; break;
            case MatrixDestination::ampLevel: voiceMod.ampLevel += value * 0.85f; break;
            case MatrixDestination::filterBalance: sampleModSums.filterBalance += value * 0.75f; break;
            case MatrixDestination::oscVolume: voiceMod.oscVolume += value * 0.8f; break;
            case MatrixDestination::subOscVolume: voiceMod.subOscVolume += value * 0.6f; break;
            case MatrixDestination::noiseVolume: voiceMod.noiseVolume += value * 0.45f; break;
            case MatrixDestination::fxMix: sampleModSums.fxMix += value * 0.5f; break;
            case MatrixDestination::fxIntensity: sampleModSums.fxIntensity += value * 0.5f; break;
            case MatrixDestination::delaySend: sampleModSums.delaySend += value * 0.5f; break;
            case MatrixDestination::delayTime: sampleModSums.delayTime += value * 0.15f; break;
            case MatrixDestination::delayFeedback: sampleModSums.delayFeedback += value * 0.35f; break;
            case MatrixDestination::reverbMix: sampleModSums.reverbMix += value * 0.45f; break;
            case MatrixDestination::reverbTime: sampleModSums.reverbTime += value * 0.3f; break;
            case MatrixDestination::lowEqGain: sampleModSums.lowEqGain += value * 10.0f; break;
            case MatrixDestination::midEqGain: sampleModSums.midEqGain += value * 10.0f; break;
            case MatrixDestination::highEqGain: sampleModSums.highEqGain += value * 10.0f; break;
            case MatrixDestination::reverbDamping: sampleModSums.reverbDamping += value * 0.35f; break;
            case MatrixDestination::lowEqFreq: sampleModSums.lowEqFreq += value * 280.0f; break;
            case MatrixDestination::lowEqQ: sampleModSums.lowEqQ += value * 0.28f; break;
            case MatrixDestination::midEqFreq: sampleModSums.midEqFreq += value * 1200.0f; break;
            case MatrixDestination::midEqQ: sampleModSums.midEqQ += value * 0.8f; break;
            case MatrixDestination::highEqFreq: sampleModSums.highEqFreq += value * 2400.0f; break;
            case MatrixDestination::highEqQ: sampleModSums.highEqQ += value * 0.22f; break;
            case MatrixDestination::masterLevel: sampleModSums.masterLevel += value * 0.28f; break;
            case MatrixDestination::detune: voiceMod.detune += value * 0.08f; break;
            case MatrixDestination::syncAmount: voiceMod.syncAmount += value * 0.45f; break;
            case MatrixDestination::gateLength: voiceMod.gateLength += value * 1.2f; break;
            case MatrixDestination::filterEnvAmount: sampleModSums.filterEnvAmount += value * 0.4f; break;
            case MatrixDestination::osc2Mix: voiceMod.osc2Mix += value * 0.45f; break;
            case MatrixDestination::ringModAmount: voiceMod.ringModAmount += value * 0.4f; break;
            case MatrixDestination::assign:
            case MatrixDestination::off:
            default:
                break;
        }
    };

    auto applyLfoDestination = [&] (int destination, int assignDestination, float value)
    {
        switch (destination)
        {
            case 0: applyMatrixDestination (static_cast<int> (MatrixDestination::osc1Pitch), value); break;
            case 1: applyMatrixDestination (static_cast<int> (MatrixDestination::osc23Pitch), value); break;
            case 2: applyMatrixDestination (static_cast<int> (MatrixDestination::pulseWidth), value); break;
            case 3: applyMatrixDestination (static_cast<int> (MatrixDestination::resonance), value); break;
            case 4: applyMatrixDestination (static_cast<int> (MatrixDestination::filterGain), value); break;
            case 5: applyMatrixDestination (static_cast<int> (MatrixDestination::cutoff1), value); break;
            case 6: applyMatrixDestination (static_cast<int> (MatrixDestination::cutoff2), value); break;
            case 7: applyMatrixDestination (static_cast<int> (MatrixDestination::shape), value); break;
            case 8: applyMatrixDestination (static_cast<int> (MatrixDestination::fmAmount), value); break;
            case 9: applyMatrixDestination (static_cast<int> (MatrixDestination::panorama), value); break;
            case 10: applyMatrixDestination (assignDestination, value); break;
            default: break;
        }
    };

    applyLfoDestination (params.lfo1Destination, params.lfo1AssignDestination, lfo1ValueNow * params.lfo1Amount);
    applyLfoDestination (params.lfo2Destination, params.lfo2AssignDestination, lfo2ValueNow * params.lfo2Amount);
    applyLfoDestination (params.lfo3Destination, params.lfo3AssignDestination, lfo3ValueNow * params.lfo3Amount);

    auto sourceValueForMatrix = [&] (int source) -> float
    {
        switch (static_cast<MatrixSource> (source))
        {
            case MatrixSource::lfo1: return lfo1ValueNow;
            case MatrixSource::lfo2: return lfo2ValueNow;
            case MatrixSource::lfo3: return lfo3ValueNow;
            case MatrixSource::filterEnv: return filtEnv;
            case MatrixSource::ampEnv: return ampEnv;
            case MatrixSource::velocity: return voice.velocity;
            case MatrixSource::note: return juce::jmap (static_cast<float> (voice.midiNote), 24.0f, 96.0f, -1.0f, 1.0f);
            case MatrixSource::random:
            {
                const auto randomPhase = std::sin ((static_cast<float> (voice.midiNote) * 17.31f) + (voice.noteAge * 13.7f));
                return juce::jlimit (-1.0f, 1.0f, randomPhase);
            }
            case MatrixSource::off:
            default:
                return 0.0f;
        }
    };

    for (const auto& slot : params.modulationMatrix)
    {
        const auto sourceValue = sourceValueForMatrix (slot.source);
        if (std::abs (sourceValue) < 0.00001f)
            continue;

        for (const auto& target : slot.targets)
        {
            if (std::abs (target.amount) < 0.00001f)
                continue;
            applyMatrixDestination (target.destination, sourceValue * target.amount);
        }
    }

    auto baseHz = midiToHz (voice.midiNote);
    baseHz *= std::pow (2.0f, voiceMod.osc1Pitch / 12.0f);

    const auto fm = fmOperator (voice, baseHz, juce::jmax (0.0f, params.fmAmount + voiceMod.fmAmount));

    const auto osc1Shape = juce::jlimit (0.05f, 0.95f, params.osc1PulseWidth + voiceMod.pulseWidth + voiceMod.shape);

    float s = 0.0f;
    for (int i = 0; i < params.unisonVoices; ++i)
    {
        const auto spread = (static_cast<float> (i) - (params.unisonVoices - 1) * 0.5f)
                            * juce::jlimit (0.0f, 0.25f, params.detune + voiceMod.detune);
        const auto osc1Pitch = params.osc1Semitone + params.osc1Detune + spread;
        s += oscSampleForState (voice.unisonPhases[static_cast<size_t> (i)],
                                voice.unisonSyncPhases[static_cast<size_t> (i)],
                                voice.unisonSamplePositions[static_cast<size_t> (i)],
                                baseHz * std::pow (2.0f, osc1Pitch / 12.0f) + fm,
                                params.oscType,
                                juce::jlimit (0.0f, 4.0f, params.syncAmount + voiceMod.syncAmount),
                                osc1Shape);
    }
    s /= static_cast<float> (params.unisonVoices);
    voice.phase = voice.unisonPhases[0];
    voice.syncPhase = voice.unisonSyncPhases[0];
    voice.samplePos = voice.unisonSamplePositions[0];

    if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + (baseHz * 0.5f) / static_cast<float> (currentSampleRate), 1.0f);
        const auto sub = std::sin (twoPi * voice.auxPhase);
        const auto squareSub = voice.auxPhase < 0.5f ? 1.0f : -1.0f;
        s = (s * 0.76f) + (sub * 0.18f) + (squareSub * 0.08f);
        s = smoothTowards (s, 0.11f, voice.toneState);
        s = softSaturate (s * 1.18f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 0.12f / static_cast<float> (currentSampleRate), 1.0f);
        const auto driftA = std::sin (twoPi * voice.auxPhase) * 0.006f;
        const auto driftB = std::sin (twoPi * std::fmod (voice.auxPhase + 0.31f, 1.0f)) * 0.008f;
        const auto ensembleA = std::sin (twoPi * std::fmod (voice.phase + driftA, 1.0f));
        const auto ensembleB = std::sin (twoPi * std::fmod ((voice.phase * 1.004f) + driftB + 0.17f, 1.0f));
        s = (s * 0.74f) + (ensembleA * 0.14f) + (ensembleB * 0.12f);
        s = smoothTowards (s, 0.04f, voice.toneState);
        s = softSaturate (s * 1.08f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 4.6f / static_cast<float> (currentSampleRate), 1.0f);
        const auto vibrato = std::sin (twoPi * voice.auxPhase) * 0.006f;
        const auto octave = std::sin (twoPi * std::fmod ((voice.phase * 2.0f) + vibrato, 1.0f));
        const auto bite = std::exp (-voice.noteAge * 4.0f);
        s = (s * 0.82f) + (octave * 0.12f * bite);
        s = softSaturate (s * 1.2f) * 0.92f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        voice.auxPhase = std::fmod (voice.auxPhase + 0.08f / static_cast<float> (currentSampleRate), 1.0f);
        const auto driftA = std::sin (twoPi * voice.auxPhase) * 0.007f;
        const auto driftB = std::sin (twoPi * std::fmod (voice.auxPhase + 0.31f, 1.0f)) * 0.01f;
        const auto wideA = std::sin (twoPi * std::fmod (voice.phase + driftA, 1.0f));
        const auto wideB = std::sin (twoPi * std::fmod ((voice.phase * 0.5f) + driftB + 0.23f, 1.0f));
        s = (s * 0.72f) + (wideA * 0.12f) + (wideB * 0.1f);
        s = smoothTowards (s, 0.03f, voice.toneState);
        s = softSaturate (s * 1.04f) * 0.88f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        const auto attackNoise = (random.nextFloat() * 2.0f - 1.0f) * 0.08f * std::exp (-voice.noteAge * 80.0f);
        const auto body = std::sin (twoPi * std::fmod ((voice.phase * 1.01f) + 0.13f, 1.0f)) * 0.12f * std::exp (-voice.noteAge * 2.8f);
        s = (s * 0.8f) + body + attackNoise;
        s = smoothTowards (s, 0.12f, voice.toneState);
        s = softSaturate (s * 1.16f) * 0.9f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
    {
        const auto glue = smoothTowards (s, 0.09f, voice.toneState);
        const auto body = std::sin (twoPi * std::fmod (voice.phase * 0.5f, 1.0f)) * 0.08f;
        s = (s * 0.88f) + (glue * 0.1f) + body;
        s = softSaturate (s * 1.08f) * 0.95f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
    {
        const auto saw = (2.0f * voice.phase) - 1.0f;
        const auto square = voice.phase < 0.5f ? 1.0f : -1.0f;
        const auto waveform = params.oscType == OscType::square ? square : saw;
        const auto squelch = std::sin (twoPi * std::fmod ((voice.phase * 2.05f) + (voice.fmPhase * 0.07f), 1.0f));
        const auto accent = juce::jlimit (0.75f, 1.35f, 0.75f + (voice.velocity * 0.85f));
        const auto notePush = std::exp (-voice.noteAge * 7.2f);
        const auto grip = juce::jmap (juce::jlimit (0.0f, 1.0f, params.fmAmount / 1000.0f), 0.14f, 0.62f);
        s = (waveform * 0.86f) + (squelch * grip * notePush);
        s += std::sin (twoPi * std::fmod (voice.phase * 0.5f, 1.0f)) * 0.05f;
        s = smoothTowards (s, 0.14f, voice.toneState);
        s = softSaturate (s * (1.45f + (params.resonance * 0.42f))) * 0.92f * accent;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        const auto osc2Shape = juce::jlimit (0.05f, 0.95f, params.osc2PulseWidth + voiceMod.pulseWidth + voiceMod.shape);
        const auto osc3Shape = juce::jlimit (0.05f, 0.95f, params.osc3PulseWidth + (voiceMod.shape * 0.75f));
        const auto osc2Ratio = std::pow (2.0f, (params.osc2Semitone + params.osc2Detune + voiceMod.osc23Pitch) / 12.0f);
        const auto osc2Hz = juce::jlimit (20.0f, 18000.0f, baseHz * osc2Ratio);
        const auto osc3Ratio = std::pow (2.0f, (params.osc3Semitone + params.osc3Detune + voiceMod.osc23Pitch) / 12.0f);
        const auto osc3Hz = juce::jlimit (10.0f, 12000.0f, baseHz * osc3Ratio);
        const auto osc2 = basicOscSample (voice.osc2Phase, osc2Hz, static_cast<OscType> (params.osc2Type), osc2Shape);
        const auto sub = basicOscSample (voice.subPhase, osc3Hz, static_cast<OscType> (params.osc3Type), osc3Shape);
        const auto ring = (s * osc2) * juce::jlimit (0.0f, 1.0f, params.ringModAmount + voiceMod.ringModAmount);
        const auto noiseBed = (random.nextFloat() * 2.0f - 1.0f) * juce::jlimit (0.0f, 1.2f, params.noiseLevel + voiceMod.noiseVolume);
        const auto oscBlend = juce::jlimit (0.0f, 1.0f, params.osc2Mix + voiceMod.osc2Mix);
        const auto fmGrip = juce::jlimit (0.0f, 1.0f, (params.fmAmount + voiceMod.fmAmount) / 1000.0f);
        const auto syncEdge = std::sin (twoPi * std::fmod ((voice.phase * (2.0f + juce::jlimit (0.0f, 4.0f, params.syncAmount + voiceMod.syncAmount) * 0.8f))
                                                           + (voice.osc2Phase * 0.37f), 1.0f));
        const auto oscDrive = juce::jlimit (0.0f, 2.0f, 1.0f + voiceMod.oscVolume);
        const auto filterGain = juce::jlimit (0.2f, 2.2f, 1.0f + voiceMod.filterGain);
        const auto subLevel = juce::jlimit (0.0f, 1.2f, params.subOscLevel + voiceMod.subOscVolume);

        s = (s * (1.0f - oscBlend * 0.75f)) + (osc2 * oscBlend);
        s *= oscDrive;
        s += (sub * (params.osc3Enabled ? subLevel : 0.0f)) + noiseBed + ring + (syncEdge * fmGrip * 0.22f);
        s *= filterGain;
        s = smoothTowards (s, 0.08f + (params.reverbMix * 0.02f), voice.toneState);
        const auto driven = s * (1.2f + (params.fxIntensity * 0.3f));
        switch (juce::jlimit (0, 3, params.saturationType))
        {
            case 1:
                s = std::tanh (driven * 0.92f) * 0.94f;
                break;
            case 2:
                s = std::atan (driven * 1.35f) * 0.78f;
                break;
            case 3:
            {
                const auto folded = std::abs (std::fmod (driven + 1.0f, 4.0f) - 2.0f) - 1.0f;
                s = softSaturate (folded * 1.05f) * 0.9f;
                break;
            }
            default:
                s = softSaturate (driven) * 0.94f;
                break;
        }
    }
    else
    {
        s = softSaturate (s * 1.12f) * 0.94f;
    }

    sampleModSums.filterEnvPeak = juce::jmax (sampleModSums.filterEnvPeak, filtEnv);
    sampleModSums.ampEnvPeak = juce::jmax (sampleModSums.ampEnvPeak, ampEnv);
    sampleModSums.notePitch += (static_cast<float> (voice.midiNote) - 60.0f) / 12.0f;
    ++sampleModSums.activeVoices;

    voice.noteAge += 1.0f / static_cast<float> (currentSampleRate);
    const auto gatePass = voice.noteAge < juce::jlimit (0.01f, 8.0f, params.gateLength + voiceMod.gateLength) ? 1.0f : 0.0f;

    const auto gatePhase = params.lfo3EnvMode ? std::fmod (params.rhythmGateRate * voice.noteAge, 1.0f) : lfo3Phase;
    const auto rg = 0.5f * (1.0f + lfoValue (params.lfo3Shape, gatePhase));
    const auto rhythmGate = 1.0f - params.rhythmGateDepth + params.rhythmGateDepth * rg;

    if (! voice.ampEnv.isActive())
        voice.active = false;

    return s * ampEnv * juce::jlimit (0.0f, 2.0f, 1.0f + voiceMod.ampLevel) * gatePass * rhythmGate * voice.velocity;
}

void AdvancedVSTiAudioProcessor::advanceArpIfNeeded()
{
    if (heldNotes.isEmpty())
        return;

    const auto mode = renderParams.arpMode;
    if (mode == 3)
    {
        arpStep = random.nextInt (heldNotes.size());
        return;
    }

    ++arpStep;
    if (arpStep >= heldNotes.size())
        arpStep = 0;
}

int AdvancedVSTiAudioProcessor::getArpNote() const
{
    if (heldNotes.isEmpty())
        return -1;

    const auto mode = renderParams.arpMode;
    if (mode == 1)
        return heldNotes[heldNotes.size() - 1 - juce::jlimit (0, heldNotes.size() - 1, arpStep)];

    return heldNotes[juce::jlimit (0, heldNotes.size() - 1, arpStep)];
}

void AdvancedVSTiAudioProcessor::updateRenderParameters()
{
    renderParams.oscType = static_cast<OscType> (paramIndex (apvts, "OSCTYPE"));
    if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
        renderParams.polyphony = juce::jlimit (1, voiceLimitForFlavor(), paramIndex (apvts, "POLYPHONY"));
    else
        renderParams.polyphony = voiceLimitForFlavor();
    renderParams.unisonVoices = juce::jlimit (1, maxUnisonForFlavor(), paramIndex (apvts, "UNISON"));
    renderParams.lfo1Shape = paramIndex (apvts, "LFO1SHAPE");
    renderParams.lfo2Shape = paramIndex (apvts, "LFO2SHAPE");
    renderParams.lfo3Shape = 0;
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        renderParams.lfo3Shape = paramIndex (apvts, "LFO3SHAPE");
    renderParams.arpMode = paramIndex (apvts, "ARPMODE");
    renderParams.arpEnabled = false;
    renderParams.lfo1Enabled = true;
    renderParams.lfo2Enabled = true;
    renderParams.lfo3Enabled = true;
    renderParams.lfo1EnvMode = false;
    renderParams.lfo2EnvMode = false;
    renderParams.lfo3EnvMode = true;
    renderParams.filter1Enabled = true;
    renderParams.filter2Enabled = true;
    renderParams.filterSlope = juce::jlimit (0, 2, paramIndex (apvts, "FILTERSLOPE"));
    renderParams.osc2Type = paramIndex (apvts, "OSC2TYPE");
    renderParams.filter2Type = paramIndex (apvts, "FILTER2TYPE");
    renderParams.filter2Slope = juce::jlimit (0, 2, paramIndex (apvts, "FILTER2SLOPE"));
    renderParams.fxType = paramIndex (apvts, "FXTYPE");
    renderParams.osc3Type = static_cast<int> (OscType::square);
    renderParams.masterLevel = 1.0f;
    if constexpr (isDrumFlavor())
        renderParams.filterType = juce::jlimit (0, 4, paramIndex (apvts, "FILTERTYPE"));
    else
        renderParams.filterType = juce::jlimit (0, 3, paramIndex (apvts, "FILTERTYPE"));

    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        renderParams.masterLevel = paramValue (apvts, "MASTERLEVEL");

    renderParams.detune = paramValue (apvts, "DETUNE");
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        const auto fmEnabled = paramValue (apvts, "FMENABLE") >= 0.5f;
        const auto syncEnabled = paramValue (apvts, "SYNCENABLE") >= 0.5f;
        const auto ringModEnabled = paramValue (apvts, "RINGMODENABLE") >= 0.5f;
        renderParams.fmAmount = fmEnabled ? paramValue (apvts, "FMAMOUNT") : 0.0f;
        renderParams.syncAmount = syncEnabled ? paramValue (apvts, "SYNC") : 0.0f;
        renderParams.ringModAmount = ringModEnabled ? paramValue (apvts, "RINGMOD") : 0.0f;
    }
    else
    {
        renderParams.fmAmount = paramValue (apvts, "FMAMOUNT");
        renderParams.syncAmount = paramValue (apvts, "SYNC");
        renderParams.ringModAmount = paramValue (apvts, "RINGMOD");
    }
    renderParams.gateLength = paramValue (apvts, "OSCGATE");
    renderParams.osc1Semitone = 0.0f;
    renderParams.osc1Detune = 0.0f;
    renderParams.osc1PulseWidth = 0.5f;
    renderParams.osc2Semitone = paramValue (apvts, "OSC2SEMITONE");
    renderParams.osc2Detune = paramValue (apvts, "OSC2DETUNE");
    renderParams.osc2PulseWidth = 0.5f;
    renderParams.osc3Semitone = -12.0f;
    renderParams.osc3Detune = 0.0f;
    renderParams.osc3PulseWidth = 0.5f;
    renderParams.osc2Mix = paramValue (apvts, "OSC2MIX");
    renderParams.subOscLevel = paramValue (apvts, "SUBOSCLEVEL");
    renderParams.osc3Enabled = true;
    renderParams.monoEnabled = false;
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        renderParams.arpEnabled = paramValue (apvts, "ARPENABLE") >= 0.5f;
        renderParams.lfo1Enabled = paramValue (apvts, "LFO1ENABLE") >= 0.5f;
        renderParams.lfo2Enabled = paramValue (apvts, "LFO2ENABLE") >= 0.5f;
        renderParams.lfo3Enabled = paramValue (apvts, "LFO3ENABLE") >= 0.5f;
        renderParams.lfo1EnvMode = paramValue (apvts, "LFO1ENVMODE") >= 0.5f;
        renderParams.lfo2EnvMode = paramValue (apvts, "LFO2ENVMODE") >= 0.5f;
        renderParams.lfo3EnvMode = paramValue (apvts, "LFO3ENVMODE") >= 0.5f;
        renderParams.filter1Enabled = paramValue (apvts, "FILTER1ENABLE") >= 0.5f;
        renderParams.filter2Enabled = paramValue (apvts, "FILTER2ENABLE") >= 0.5f;
        renderParams.osc3Type = paramIndex (apvts, "OSC3TYPE");
        renderParams.osc3Enabled = paramValue (apvts, "OSC3ENABLE") >= 0.5f;
        renderParams.monoEnabled = paramValue (apvts, "MONOENABLE") >= 0.5f;
        renderParams.osc1Semitone = paramValue (apvts, "OSC1SEMITONE");
        renderParams.osc1Detune = paramValue (apvts, "OSC1DETUNE");
        renderParams.osc1PulseWidth = paramValue (apvts, "OSC1PW");
        renderParams.osc2PulseWidth = paramValue (apvts, "OSC2PW");
        renderParams.osc3Semitone = paramValue (apvts, "OSC3SEMITONE");
        renderParams.osc3Detune = paramValue (apvts, "OSC3DETUNE");
        renderParams.osc3PulseWidth = paramValue (apvts, "OSC3PW");
    }
    renderParams.noiseLevel = paramValue (apvts, "NOISELEVEL");
    renderParams.envCurve = paramValue (apvts, "ENVCURVE");
    renderParams.cutoff = paramValue (apvts, "CUTOFF");
    renderParams.cutoff2 = paramValue (apvts, "CUTOFF2");
    renderParams.resonance = paramValue (apvts, "RESONANCE");
    renderParams.resonance2 = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "RESONANCE2") : renderParams.resonance;
    renderParams.filterEnvAmount = paramValue (apvts, "FILTERENVAMOUNT");
    renderParams.filterBalance = paramValue (apvts, "FILTERBALANCE");
    renderParams.panorama = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "PANORAMA") : 0.0f;
    renderParams.keyFollow = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "KEYFOLLOW") : 0.0f;
    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
    {
        renderParams.sampleBank = paramIndex (apvts, "SAMPLEBANK");
        renderParams.sampleStart = paramValue (apvts, "SAMPLESTART");
        renderParams.sampleEnd = juce::jlimit (0.02f, 1.0f, paramValue (apvts, "SAMPLEEND"));
        if (renderParams.sampleEnd <= renderParams.sampleStart)
            renderParams.sampleEnd = juce::jlimit (0.02f, 1.0f, renderParams.sampleStart + 0.02f);
    }
    else
    {
        renderParams.sampleBank = 0;
        renderParams.sampleStart = 0.0f;
        renderParams.sampleEnd = 1.0f;
    }
    renderParams.lfo1Rate = paramValue (apvts, "LFO1RATE");
    renderParams.lfo1Amount = renderParams.lfo1Enabled
                                  ? (buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "LFO1AMOUNT")
                                                                                : juce::jlimit (-1.0f, 1.0f, paramValue (apvts, "LFO1PITCH") / 24.0f))
                                  : 0.0f;
    renderParams.lfo1Destination = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "LFO1DEST") : 0;
    renderParams.lfo1AssignDestination = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "LFO1ASSIGNDEST")
                                                                                     : static_cast<int> (MatrixDestination::off);
    renderParams.lfo1Pitch = renderParams.lfo1Enabled ? paramValue (apvts, "LFO1PITCH") : 0.0f;
    renderParams.lfo2Rate = paramValue (apvts, "LFO2RATE");
    renderParams.lfo2Amount = renderParams.lfo2Enabled
                                  ? (buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "LFO2AMOUNT")
                                                                                : juce::jlimit (-1.0f, 1.0f, paramValue (apvts, "LFO2FILTER")))
                                  : 0.0f;
    renderParams.lfo2Destination = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "LFO2DEST") : 5;
    renderParams.lfo2AssignDestination = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "LFO2ASSIGNDEST")
                                                                                     : static_cast<int> (MatrixDestination::off);
    renderParams.lfo2Filter = renderParams.lfo2Enabled ? paramValue (apvts, "LFO2FILTER") : 0.0f;
    renderParams.arpRate = paramValue (apvts, "ARPRATE");
    renderParams.rhythmGateRate = paramValue (apvts, "RHYTHMGATE_RATE");
    renderParams.lfo3Amount = renderParams.lfo3Enabled
                                  ? (buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "LFO3AMOUNT")
                                                                                : paramValue (apvts, "RHYTHMGATE_DEPTH"))
                                  : 0.0f;
    renderParams.lfo3Destination = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "LFO3DEST") : 10;
    renderParams.lfo3AssignDestination = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "LFO3ASSIGNDEST")
                                                                                     : static_cast<int> (MatrixDestination::ampLevel);
    renderParams.rhythmGateDepth = renderParams.lfo3Enabled ? paramValue (apvts, "RHYTHMGATE_DEPTH") : 0.0f;
    renderParams.fxMix = paramValue (apvts, "FXMIX");
    renderParams.fxIntensity = paramValue (apvts, "FXINTENSITY");
    renderParams.fxRate = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "FXRATE") : 0.65f;
    renderParams.fxColour = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "FXCOLOUR") : 0.5f;
    renderParams.fxSpread = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "FXSPREAD") : 0.4f;
    renderParams.delaySend = paramValue (apvts, "DELAYSEND");
    renderParams.delayTimeSec = paramValue (apvts, "DELAYTIME");
    renderParams.delayFeedback = paramValue (apvts, "DELAYFEEDBACK");
    renderParams.reverbMix = paramValue (apvts, "REVERBMIX");
    renderParams.reverbTime = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "REVERBTIME") : 0.45f;
    renderParams.reverbDamping = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "REVERBDAMP") : 0.45f;
    renderParams.lowEqGainDb = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "LOWEQGAIN") : 0.0f;
    renderParams.lowEqFreq = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "LOWEQFREQ") : 220.0f;
    renderParams.lowEqQ = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "LOWEQQ") : 0.8f;
    renderParams.midEqGainDb = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "MIDEQGAIN") : 0.0f;
    renderParams.midEqFreq = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "MIDEQFREQ") : 1400.0f;
    renderParams.midEqQ = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "MIDEQQ") : 1.1f;
    renderParams.highEqGainDb = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "HIGHEQGAIN") : 0.0f;
    renderParams.highEqFreq = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "HIGHEQFREQ") : 5000.0f;
    renderParams.highEqQ = buildFlavor() == InstrumentFlavor::advanced ? paramValue (apvts, "HIGHEQQ") : 0.8f;
    renderParams.saturationType = buildFlavor() == InstrumentFlavor::advanced ? paramIndex (apvts, "SATURATIONTYPE") : 0;

    for (auto& slot : renderParams.modulationMatrix)
    {
        slot.source = static_cast<int> (MatrixSource::off);
        for (auto& target : slot.targets)
        {
            target.destination = static_cast<int> (MatrixDestination::off);
            target.amount = 0.0f;
        }
    }

    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        for (int slotIndex = 0; slotIndex < 6; ++slotIndex)
        {
            auto& slot = renderParams.modulationMatrix[static_cast<size_t> (slotIndex)];
            const auto slotId = juce::String (slotIndex + 1);
            slot.source = paramIndex (apvts, "MATRIX" + slotId + "SOURCE");
            for (int targetIndex = 0; targetIndex < 3; ++targetIndex)
            {
                auto& target = slot.targets[static_cast<size_t> (targetIndex)];
                const auto targetId = juce::String (targetIndex + 1);
                target.amount = paramValue (apvts, "MATRIX" + slotId + "AMOUNT" + targetId);
                target.destination = paramIndex (apvts, "MATRIX" + slotId + "DEST" + targetId);
            }
        }
    }

    if constexpr (isSynthDrumFlavor())
    {
        renderParams.drumMasterLevel = paramValue (apvts, "DRUMMASTERLEVEL");
        renderParams.drumKickAttack = paramValue (apvts, "DRUM_KICK_ATTACK");
        renderParams.drumSnareTone = paramValue (apvts, "DRUM_SNARE_TONE");
        renderParams.drumSnareSnappy = paramValue (apvts, "DRUM_SNARE_SNAPPY");
        renderParams.drumVoiceTunes.fill (0.0f);
        renderParams.drumVoiceDecays.fill (0.5f);
        renderParams.externalPadLevels.fill (1.0f);
        renderParams.externalPadSustainTimes.fill (120.0f);
        renderParams.externalPadReleaseTimes.fill (0.2f);

        for (size_t index = 0; index < drumVoiceTuneParamIds.size(); ++index)
            renderParams.drumVoiceTunes[drumVoiceIndex (drumVoiceTuneKinds[index])] = paramValue (apvts, drumVoiceTuneParamIds[index]);

        for (size_t index = 0; index < drumVoiceDecayParamIds.size(); ++index)
            renderParams.drumVoiceDecays[drumVoiceIndex (drumVoiceDecayKinds[index])] = paramValue (apvts, drumVoiceDecayParamIds[index]);

        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
            renderParams.drumVoiceLevels[index] = paramValue (apvts, drumVoiceLevelParamIds[index]);
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        renderParams.drumMasterLevel = paramValue (apvts, "DRUMMASTERLEVEL");
        renderParams.drumKickAttack = 0.5f;
        renderParams.drumSnareTone = 0.5f;
        renderParams.drumSnareSnappy = 0.5f;
        renderParams.drumVoiceTunes.fill (0.0f);
        renderParams.drumVoiceDecays.fill (0.5f);
        renderParams.drumVoiceLevels.fill (1.0f);
        renderParams.externalPadSustainTimes.fill (120.0f);
        renderParams.externalPadReleaseTimes.fill (0.2f);

        for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
        {
            renderParams.externalPadLevels[static_cast<size_t> (padIndex)] = paramValue (apvts, externalPadLevelParameterIdForIndex (padIndex).toRawUTF8());
            renderParams.externalPadSustainTimes[static_cast<size_t> (padIndex)] = paramValue (apvts, externalPadSustainParameterIdForIndex (padIndex).toRawUTF8());
            renderParams.externalPadReleaseTimes[static_cast<size_t> (padIndex)] = paramValue (apvts, externalPadReleaseParameterIdForIndex (padIndex).toRawUTF8());
        }
    }
    else
    {
        renderParams.drumMasterLevel = 1.0f;
        renderParams.drumKickAttack = 0.5f;
        renderParams.drumSnareTone = 0.5f;
        renderParams.drumSnareSnappy = 0.5f;
        renderParams.drumVoiceTunes.fill (0.0f);
        renderParams.drumVoiceDecays.fill (0.5f);
        renderParams.drumVoiceLevels.fill (1.0f);
        renderParams.externalPadLevels.fill (1.0f);
        renderParams.externalPadSustainTimes.fill (120.0f);
        renderParams.externalPadReleaseTimes.fill (0.2f);
    }

    renderParams.ampEnv.attack = paramValue (apvts, "AMPATTACK");
    renderParams.ampEnv.decay = paramValue (apvts, "AMPDECAY");
    renderParams.ampEnv.sustain = paramValue (apvts, "AMPSUSTAIN");
    renderParams.ampEnv.release = paramValue (apvts, "AMPRELEASE");

    renderParams.filterEnv.attack = paramValue (apvts, "FILTATTACK");
    renderParams.filterEnv.decay = paramValue (apvts, "FILTDECAY");
    renderParams.filterEnv.sustain = paramValue (apvts, "FILTSUSTAIN");
    renderParams.filterEnv.release = paramValue (apvts, "FILTRELEASE");

    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
        refreshSampleBank();
}

void AdvancedVSTiAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    buffer.clear();

    updateRenderParameters();
    applyEnvelopeSettings();

    {
        const juce::SpinLock::ScopedLockType lock (pendingPreviewMidiLock);
        midiMessages.addEvents (pendingPreviewMidi, 0, -1, 0);
        pendingPreviewMidi.clear();
    }

    applyPendingUiActions (midiMessages, buffer.getNumSamples());
    keyboardState.processNextMidiBuffer (midiMessages, 0, buffer.getNumSamples(), true);

    leftFilter.setResonance (renderParams.resonance);
    rightFilter.setResonance (renderParams.resonance);
    leftFilterCascade.setResonance (renderParams.resonance);
    rightFilterCascade.setResonance (renderParams.resonance);
    leftFilter2.setResonance (renderParams.resonance);
    rightFilter2.setResonance (renderParams.resonance);
    leftFilter2Cascade.setResonance (renderParams.resonance);
    rightFilter2Cascade.setResonance (renderParams.resonance);

    chorusLeft.setRate (0.12f + renderParams.fxIntensity * 1.8f);
    chorusRight.setRate (0.14f + renderParams.fxIntensity * 1.7f);
    chorusLeft.setDepth (0.1f + renderParams.fxIntensity * 0.75f);
    chorusRight.setDepth (0.12f + renderParams.fxIntensity * 0.72f);
    chorusLeft.setCentreDelay (6.0f + renderParams.fxIntensity * 12.0f);
    chorusRight.setCentreDelay (7.0f + renderParams.fxIntensity * 11.0f);
    chorusLeft.setFeedback (0.05f + renderParams.fxIntensity * 0.18f);
    chorusRight.setFeedback (0.04f + renderParams.fxIntensity * 0.18f);

    flangerLeft.setRate (0.04f + renderParams.fxIntensity * 0.7f);
    flangerRight.setRate (0.05f + renderParams.fxIntensity * 0.75f);
    flangerLeft.setDepth (0.2f + renderParams.fxIntensity * 0.78f);
    flangerRight.setDepth (0.22f + renderParams.fxIntensity * 0.74f);
    flangerLeft.setCentreDelay (1.2f + renderParams.fxIntensity * 3.1f);
    flangerRight.setCentreDelay (1.4f + renderParams.fxIntensity * 2.8f);
    flangerLeft.setFeedback (0.08f + renderParams.fxIntensity * 0.42f);
    flangerRight.setFeedback (0.08f + renderParams.fxIntensity * 0.42f);

    phaserLeft.setRate (0.08f + renderParams.fxIntensity * 1.3f);
    phaserRight.setRate (0.09f + renderParams.fxIntensity * 1.35f);
    phaserLeft.setDepth (0.15f + renderParams.fxIntensity * 0.72f);
    phaserRight.setDepth (0.17f + renderParams.fxIntensity * 0.68f);
    phaserLeft.setCentreFrequency (400.0f + renderParams.fxIntensity * 1600.0f);
    phaserRight.setCentreFrequency (520.0f + renderParams.fxIntensity * 1500.0f);
    phaserLeft.setFeedback (0.05f + renderParams.fxIntensity * 0.25f);
    phaserRight.setFeedback (0.05f + renderParams.fxIntensity * 0.25f);

    juce::Reverb::Parameters reverbParams;
    reverbParams.roomSize = 0.2f + renderParams.reverbMix * 0.65f;
    reverbParams.damping = 0.25f + renderParams.fxIntensity * 0.5f;
    reverbParams.wetLevel = 1.0f;
    reverbParams.dryLevel = 0.0f;
    reverbParams.width = 0.8f;
    reverbParams.freezeMode = 0.0f;
    reverb.setParameters (reverbParams);

    bool bypassFilter = false;
    if constexpr (isDrumFlavor())
    {
        bypassFilter = renderParams.filterType == 0;
        if (! bypassFilter)
        {
            const auto filterMode = juce::jlimit (0, 3, renderParams.filterType - 1);
            leftFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterMode));
            rightFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterMode));
            leftFilterCascade.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterMode));
            rightFilterCascade.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (filterMode));
        }
    }
    else
    {
        leftFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
        rightFilter.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
        leftFilterCascade.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
        rightFilterCascade.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (renderParams.filterType));
        leftFilter2.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (juce::jlimit (0, 3, renderParams.filter2Type)));
        rightFilter2.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (juce::jlimit (0, 3, renderParams.filter2Type)));
        leftFilter2Cascade.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (juce::jlimit (0, 3, renderParams.filter2Type)));
        rightFilter2Cascade.setType (static_cast<juce::dsp::StateVariableTPTFilterType> (juce::jlimit (0, 3, renderParams.filter2Type)));
    }

    int arpCounter = 0;
    const auto arpIntervalSamples = juce::jmax (1, static_cast<int> (currentSampleRate / juce::jmax (0.25f, renderParams.arpRate)));
    float blockFxMixMod = 0.0f;
    float blockFxIntensityMod = 0.0f;
    float blockDelaySendMod = 0.0f;
    float blockDelayTimeMod = 0.0f;
    float blockDelayFeedbackMod = 0.0f;
    float blockReverbMixMod = 0.0f;
    float blockReverbTimeMod = 0.0f;
    float blockReverbDampingMod = 0.0f;
    float blockLowEqGainMod = 0.0f;
    float blockLowEqFreqMod = 0.0f;
    float blockLowEqQMod = 0.0f;
    float blockMidEqGainMod = 0.0f;
    float blockMidEqFreqMod = 0.0f;
    float blockMidEqQMod = 0.0f;
    float blockHighEqGainMod = 0.0f;
    float blockHighEqFreqMod = 0.0f;
    float blockHighEqQMod = 0.0f;
    float blockMasterLevelMod = 0.0f;
    juce::MidiBuffer::Iterator midiIterator (midiMessages);
    juce::MidiMessage nextMidiMessage;
    int nextMidiSample = 0;
    auto hasMidi = midiIterator.getNextEvent (nextMidiMessage, nextMidiSample);

    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        while (hasMidi && nextMidiSample <= sample)
        {
            handleMidiMessage (nextMidiMessage);
            hasMidi = midiIterator.getNextEvent (nextMidiMessage, nextMidiSample);
        }

        if (renderParams.arpEnabled && ++arpCounter >= arpIntervalSamples)
        {
            arpCounter = 0;
            advanceArpIfNeeded();
            const auto arpNote = getArpNote();
            if (arpNote >= 0)
            {
                for (auto& voice : voices)
                {
                    if (voice.active)
                    {
                        voice.midiNote = arpNote;
                        break;
                    }
                }
            }
        }

        SampleModulationSums sampleModSums;
        float sum = 0.0f;
        const auto voiceLimit = activeVoiceLimit();
        for (int voiceIndex = 0; voiceIndex < voiceLimit; ++voiceIndex)
            sum += renderVoiceSample (voices[static_cast<size_t> (voiceIndex)], sampleModSums);

        const auto activeVoicesThisSample = juce::jmax (1, sampleModSums.activeVoices);
        currentFilterEnvPeak = sampleModSums.filterEnvPeak;

        const auto resonanceMod = sampleModSums.resonance / static_cast<float> (activeVoicesThisSample);
        const auto dynamicResonance = juce::jlimit (0.1f, 12.0f, renderParams.resonance + resonanceMod);
        const auto dynamicResonance2 = juce::jlimit (0.1f, 12.0f, renderParams.resonance2 + resonanceMod);
        leftFilter.setResonance (dynamicResonance);
        rightFilter.setResonance (dynamicResonance);
        leftFilterCascade.setResonance (dynamicResonance);
        rightFilterCascade.setResonance (dynamicResonance);
        leftFilter2.setResonance (dynamicResonance2);
        rightFilter2.setResonance (dynamicResonance2);
        leftFilter2Cascade.setResonance (dynamicResonance2);
        rightFilter2Cascade.setResonance (dynamicResonance2);

        const auto dynamicFilterEnvAmount = juce::jlimit (0.0f,
                                                          1.5f,
                                                          renderParams.filterEnvAmount
                                                              + (sampleModSums.filterEnvAmount / static_cast<float> (activeVoicesThisSample)));

        const auto averageNotePitch = sampleModSums.notePitch / static_cast<float> (activeVoicesThisSample);
        const auto keyFollowRatio = std::pow (2.0f, averageNotePitch * renderParams.keyFollow);

        auto cutoff = renderParams.cutoff;
        cutoff += currentFilterEnvPeak * dynamicFilterEnvAmount * 10000.0f;
        cutoff += sampleModSums.cutoff1 / static_cast<float> (activeVoicesThisSample);
        cutoff *= keyFollowRatio;
        cutoff = juce::jlimit (20.0f, 20000.0f, cutoff);
        leftFilter.setCutoffFrequency (cutoff);
        rightFilter.setCutoffFrequency (cutoff);
        leftFilterCascade.setCutoffFrequency (cutoff);
        rightFilterCascade.setCutoffFrequency (cutoff);

        auto cutoff2 = renderParams.cutoff2;
        cutoff2 += currentFilterEnvPeak * dynamicFilterEnvAmount * 6500.0f;
        cutoff2 += sampleModSums.cutoff2 / static_cast<float> (activeVoicesThisSample);
        cutoff2 *= keyFollowRatio;
        cutoff2 = juce::jlimit (20.0f, 20000.0f, cutoff2);
        leftFilter2.setCutoffFrequency (cutoff2);
        rightFilter2.setCutoffFrequency (cutoff2);
        leftFilter2Cascade.setCutoffFrequency (cutoff2);
        rightFilter2Cascade.setCutoffFrequency (cutoff2);

        auto processSlope = [] (auto& firstStage, auto& secondStage, float input, int slopeIndex) -> float
        {
            const auto firstStageOutput = firstStage.processSample (0, input);
            if (slopeIndex <= 0)
                return firstStageOutput;

            const auto secondStageOutput = secondStage.processSample (0, firstStageOutput);
            return juce::jmap (filterCascadeBlendForSlope (slopeIndex), firstStageOutput, secondStageOutput);
        };

        const auto filteredL = (bypassFilter || ! renderParams.filter1Enabled) ? sum : processSlope (leftFilter, leftFilterCascade, sum, renderParams.filterSlope);
        const auto filteredR = (bypassFilter || ! renderParams.filter1Enabled) ? sum : processSlope (rightFilter, rightFilterCascade, sum, renderParams.filterSlope);
        auto mixedL = filteredL;
        auto mixedR = filteredR;
        if constexpr (! isDrumFlavor())
        {
            const auto filtered2L = renderParams.filter2Enabled ? processSlope (leftFilter2, leftFilter2Cascade, sum, renderParams.filter2Slope) : sum;
            const auto filtered2R = renderParams.filter2Enabled ? processSlope (rightFilter2, rightFilter2Cascade, sum, renderParams.filter2Slope) : sum;

            if (renderParams.filter1Enabled && renderParams.filter2Enabled)
            {
                const auto balance = juce::jlimit (0.0f,
                                                   1.0f,
                                                   renderParams.filterBalance
                                                       + (sampleModSums.filterBalance / static_cast<float> (activeVoicesThisSample)));
                mixedL = juce::jmap (balance, filteredL, filtered2L);
                mixedR = juce::jmap (balance, filteredR, filtered2R);
            }
            else if (renderParams.filter2Enabled)
            {
                mixedL = filtered2L;
                mixedR = filtered2R;
            }
            else if (! renderParams.filter1Enabled)
            {
                mixedL = sum;
                mixedR = sum;
            }
        }

        const auto panorama = juce::jlimit (-1.0f,
                                            1.0f,
                                            renderParams.panorama
                                                + (sampleModSums.panorama / static_cast<float> (activeVoicesThisSample)));
        const auto panLeftGain = juce::jlimit (0.0f, 1.0f, panorama <= 0.0f ? 1.0f : 1.0f - panorama);
        const auto panRightGain = juce::jlimit (0.0f, 1.0f, panorama >= 0.0f ? 1.0f : 1.0f + panorama);
        mixedL *= panLeftGain;
        mixedR *= panRightGain;

        buffer.setSample (0, sample, mixedL);
        buffer.setSample (1, sample, mixedR);

        blockFxMixMod += sampleModSums.fxMix / static_cast<float> (activeVoicesThisSample);
        blockFxIntensityMod += sampleModSums.fxIntensity / static_cast<float> (activeVoicesThisSample);
        blockDelaySendMod += sampleModSums.delaySend / static_cast<float> (activeVoicesThisSample);
        blockDelayTimeMod += sampleModSums.delayTime / static_cast<float> (activeVoicesThisSample);
        blockDelayFeedbackMod += sampleModSums.delayFeedback / static_cast<float> (activeVoicesThisSample);
        blockReverbMixMod += sampleModSums.reverbMix / static_cast<float> (activeVoicesThisSample);
        blockReverbTimeMod += sampleModSums.reverbTime / static_cast<float> (activeVoicesThisSample);
        blockReverbDampingMod += sampleModSums.reverbDamping / static_cast<float> (activeVoicesThisSample);
        blockLowEqGainMod += sampleModSums.lowEqGain / static_cast<float> (activeVoicesThisSample);
        blockLowEqFreqMod += sampleModSums.lowEqFreq / static_cast<float> (activeVoicesThisSample);
        blockLowEqQMod += sampleModSums.lowEqQ / static_cast<float> (activeVoicesThisSample);
        blockMidEqGainMod += sampleModSums.midEqGain / static_cast<float> (activeVoicesThisSample);
        blockMidEqFreqMod += sampleModSums.midEqFreq / static_cast<float> (activeVoicesThisSample);
        blockMidEqQMod += sampleModSums.midEqQ / static_cast<float> (activeVoicesThisSample);
        blockHighEqGainMod += sampleModSums.highEqGain / static_cast<float> (activeVoicesThisSample);
        blockHighEqFreqMod += sampleModSums.highEqFreq / static_cast<float> (activeVoicesThisSample);
        blockHighEqQMod += sampleModSums.highEqQ / static_cast<float> (activeVoicesThisSample);
        blockMasterLevelMod += sampleModSums.masterLevel / static_cast<float> (activeVoicesThisSample);

        lfo1Phase = std::fmod (lfo1Phase + renderParams.lfo1Rate / static_cast<float> (currentSampleRate), 1.0f);
        lfo2Phase = std::fmod (lfo2Phase + renderParams.lfo2Rate / static_cast<float> (currentSampleRate), 1.0f);
        lfo3Phase = std::fmod (lfo3Phase + renderParams.rhythmGateRate / static_cast<float> (currentSampleRate), 1.0f);
    }

    while (hasMidi)
    {
        handleMidiMessage (nextMidiMessage);
        hasMidi = midiIterator.getNextEvent (nextMidiMessage, nextMidiSample);
    }

    if constexpr (buildFlavor() == InstrumentFlavor::advanced || buildFlavor() == InstrumentFlavor::stringSynth)
    {
        if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        {
            const auto invSamples = buffer.getNumSamples() > 0 ? 1.0f / static_cast<float> (buffer.getNumSamples()) : 0.0f;
            renderParams.fxMix = juce::jlimit (0.0f, 1.0f, renderParams.fxMix + blockFxMixMod * invSamples);
            renderParams.fxIntensity = juce::jlimit (0.0f, 1.0f, renderParams.fxIntensity + blockFxIntensityMod * invSamples);
            renderParams.delaySend = juce::jlimit (0.0f, 1.0f, renderParams.delaySend + blockDelaySendMod * invSamples);
            renderParams.delayTimeSec = juce::jlimit (0.05f, 1.2f, renderParams.delayTimeSec + blockDelayTimeMod * invSamples);
            renderParams.delayFeedback = juce::jlimit (0.0f, 0.92f, renderParams.delayFeedback + blockDelayFeedbackMod * invSamples);
            renderParams.reverbMix = juce::jlimit (0.0f, 1.0f, renderParams.reverbMix + blockReverbMixMod * invSamples);
            renderParams.reverbTime = juce::jlimit (0.1f, 1.0f, renderParams.reverbTime + blockReverbTimeMod * invSamples);
            renderParams.reverbDamping = juce::jlimit (0.0f, 1.0f, renderParams.reverbDamping + blockReverbDampingMod * invSamples);
            renderParams.lowEqGainDb = juce::jlimit (-16.0f, 16.0f, renderParams.lowEqGainDb + blockLowEqGainMod * invSamples);
            renderParams.lowEqFreq = juce::jlimit (40.0f, 1200.0f, renderParams.lowEqFreq + blockLowEqFreqMod * invSamples);
            renderParams.lowEqQ = juce::jlimit (0.3f, 2.5f, renderParams.lowEqQ + blockLowEqQMod * invSamples);
            renderParams.midEqGainDb = juce::jlimit (-16.0f, 16.0f, renderParams.midEqGainDb + blockMidEqGainMod * invSamples);
            renderParams.midEqFreq = juce::jlimit (120.0f, 8000.0f, renderParams.midEqFreq + blockMidEqFreqMod * invSamples);
            renderParams.midEqQ = juce::jlimit (0.3f, 8.0f, renderParams.midEqQ + blockMidEqQMod * invSamples);
            renderParams.highEqGainDb = juce::jlimit (-16.0f, 16.0f, renderParams.highEqGainDb + blockHighEqGainMod * invSamples);
            renderParams.highEqFreq = juce::jlimit (1200.0f, 16000.0f, renderParams.highEqFreq + blockHighEqFreqMod * invSamples);
            renderParams.highEqQ = juce::jlimit (0.3f, 2.5f, renderParams.highEqQ + blockHighEqQMod * invSamples);
            renderParams.masterLevel = juce::jlimit (0.0f, 1.5f, renderParams.masterLevel + blockMasterLevelMod * invSamples);
        }

        applyAdvancedEffects (buffer);
        if constexpr (buildFlavor() == InstrumentFlavor::advanced)
            buffer.applyGain (juce::jlimit (0.0f, 1.5f, renderParams.masterLevel));
    }
}

void AdvancedVSTiAudioProcessor::applyAdvancedEffects (juce::AudioBuffer<float>& buffer)
{
    auto* left = buffer.getWritePointer (0);
    auto* right = buffer.getWritePointer (1);
    const auto numSamples = buffer.getNumSamples();
    const auto fxMix = juce::jlimit (0.0f, 1.0f, renderParams.fxMix);
    const auto fxRate = juce::jlimit (0.02f, 12.0f, renderParams.fxRate);
    const auto fxColour = juce::jlimit (0.0f, 1.0f, renderParams.fxColour);
    const auto fxSpread = juce::jlimit (0.0f, 1.0f, renderParams.fxSpread);
    const auto delaySend = juce::jlimit (0.0f, 1.0f, renderParams.delaySend);
    const auto reverbMix = juce::jlimit (0.0f, 1.0f, renderParams.reverbMix);
    const auto feedback = juce::jlimit (0.0f, 0.95f, renderParams.delayFeedback);

    const auto chorusRate = juce::jlimit (0.05f, 6.0f, fxRate);
    const auto chorusDepth = juce::jlimit (0.05f, 0.95f, 0.18f + renderParams.fxIntensity * 0.65f + fxSpread * 0.08f);
    const auto chorusDelay = 5.0f + fxColour * 11.0f;
    const auto chorusFeedback = juce::jlimit (-0.25f, 0.6f, -0.05f + fxSpread * 0.55f);
    chorusLeft.setRate (chorusRate);
    chorusRight.setRate (chorusRate * (1.02f + fxSpread * 0.05f));
    chorusLeft.setDepth (chorusDepth);
    chorusRight.setDepth (juce::jlimit (0.05f, 0.95f, chorusDepth * (0.96f + fxSpread * 0.06f)));
    chorusLeft.setCentreDelay (chorusDelay);
    chorusRight.setCentreDelay (chorusDelay + 0.8f + fxSpread * 2.2f);
    chorusLeft.setFeedback (chorusFeedback);
    chorusRight.setFeedback (chorusFeedback * 0.92f);

    const auto flangerRate = juce::jlimit (0.03f, 3.0f, fxRate * 0.55f);
    const auto flangerDepth = juce::jlimit (0.08f, 0.98f, 0.22f + renderParams.fxIntensity * 0.72f);
    const auto flangerDelay = 0.7f + fxColour * 4.6f;
    const auto flangerFeedback = juce::jlimit (-0.85f, 0.85f, fxSpread * 1.4f - 0.7f);
    flangerLeft.setRate (flangerRate);
    flangerRight.setRate (flangerRate * (1.04f + fxSpread * 0.08f));
    flangerLeft.setDepth (flangerDepth);
    flangerRight.setDepth (juce::jlimit (0.08f, 0.98f, flangerDepth * (0.94f + fxSpread * 0.08f)));
    flangerLeft.setCentreDelay (flangerDelay);
    flangerRight.setCentreDelay (flangerDelay + 0.35f + fxSpread * 1.2f);
    flangerLeft.setFeedback (flangerFeedback);
    flangerRight.setFeedback (flangerFeedback * 0.95f);

    const auto phaserRate = juce::jlimit (0.03f, 5.0f, fxRate * 0.7f);
    const auto phaserDepth = juce::jlimit (0.05f, 0.95f, 0.18f + renderParams.fxIntensity * 0.68f);
    const auto phaserCentre = 180.0f + fxColour * 2600.0f;
    const auto phaserFeedback = juce::jlimit (-0.6f, 0.78f, fxSpread * 1.05f - 0.18f);
    phaserLeft.setRate (phaserRate);
    phaserRight.setRate (phaserRate * (1.03f + fxSpread * 0.04f));
    phaserLeft.setDepth (phaserDepth);
    phaserRight.setDepth (juce::jlimit (0.05f, 0.95f, phaserDepth * (0.97f + fxSpread * 0.05f)));
    phaserLeft.setCentreFrequency (phaserCentre);
    phaserRight.setCentreFrequency (phaserCentre * (1.08f + fxSpread * 0.08f));
    phaserLeft.setFeedback (phaserFeedback);
    phaserRight.setFeedback (phaserFeedback * 0.94f);

    juce::Reverb::Parameters reverbParams;
    reverbParams.roomSize = juce::jlimit (0.05f, 1.0f, renderParams.reverbTime);
    reverbParams.damping = juce::jlimit (0.0f, 1.0f, renderParams.reverbDamping);
    reverbParams.wetLevel = 1.0f;
    reverbParams.dryLevel = 0.0f;
    reverbParams.width = 0.8f;
    reverbParams.freezeMode = 0.0f;
    reverb.setParameters (reverbParams);

    lowEqLeft.coefficients = juce::dsp::IIR::Coefficients<float>::makeLowShelf (currentSampleRate,
                                                                                 juce::jlimit (40.0f, 1200.0f, renderParams.lowEqFreq),
                                                                                 juce::jlimit (0.3f, 2.5f, renderParams.lowEqQ),
                                                                                 juce::Decibels::decibelsToGain (renderParams.lowEqGainDb));
    lowEqRight.coefficients = juce::dsp::IIR::Coefficients<float>::makeLowShelf (currentSampleRate,
                                                                                  juce::jlimit (40.0f, 1200.0f, renderParams.lowEqFreq),
                                                                                  juce::jlimit (0.3f, 2.5f, renderParams.lowEqQ),
                                                                                  juce::Decibels::decibelsToGain (renderParams.lowEqGainDb));
    midEqLeft.coefficients = juce::dsp::IIR::Coefficients<float>::makePeakFilter (currentSampleRate,
                                                                                   juce::jlimit (120.0f, 8000.0f, renderParams.midEqFreq),
                                                                                   juce::jlimit (0.3f, 8.0f, renderParams.midEqQ),
                                                                                   juce::Decibels::decibelsToGain (renderParams.midEqGainDb));
    midEqRight.coefficients = juce::dsp::IIR::Coefficients<float>::makePeakFilter (currentSampleRate,
                                                                                    juce::jlimit (120.0f, 8000.0f, renderParams.midEqFreq),
                                                                                    juce::jlimit (0.3f, 8.0f, renderParams.midEqQ),
                                                                                    juce::Decibels::decibelsToGain (renderParams.midEqGainDb));
    highEqLeft.coefficients = juce::dsp::IIR::Coefficients<float>::makeHighShelf (currentSampleRate,
                                                                                   juce::jlimit (1200.0f, 16000.0f, renderParams.highEqFreq),
                                                                                   juce::jlimit (0.3f, 2.5f, renderParams.highEqQ),
                                                                                   juce::Decibels::decibelsToGain (renderParams.highEqGainDb));
    highEqRight.coefficients = juce::dsp::IIR::Coefficients<float>::makeHighShelf (currentSampleRate,
                                                                                    juce::jlimit (1200.0f, 16000.0f, renderParams.highEqFreq),
                                                                                    juce::jlimit (0.3f, 2.5f, renderParams.highEqQ),
                                                                                    juce::Decibels::decibelsToGain (renderParams.highEqGainDb));

    if (fxMix > 0.0001f)
    {
        if (renderParams.fxType == 3 || renderParams.fxType == 4 || renderParams.fxType == 5)
        {
            juce::AudioBuffer<float> wetBuffer;
            wetBuffer.makeCopyOf (buffer, true);
            juce::dsp::AudioBlock<float> wetBlock (wetBuffer);
            auto leftBlock = wetBlock.getSingleChannelBlock (0);
            auto rightBlock = wetBlock.getSingleChannelBlock (1);
            juce::dsp::ProcessContextReplacing<float> leftContext (leftBlock);
            juce::dsp::ProcessContextReplacing<float> rightContext (rightBlock);

            if (renderParams.fxType == 4)
            {
                phaserLeft.process (leftContext);
                phaserRight.process (rightContext);
            }
            else if (renderParams.fxType == 3)
            {
                chorusLeft.process (leftContext);
                chorusRight.process (rightContext);
            }
            else
            {
                flangerLeft.process (leftContext);
                flangerRight.process (rightContext);
            }

            for (int sample = 0; sample < numSamples; ++sample)
            {
                left[sample] = juce::jmap (fxMix, left[sample], wetBuffer.getSample (0, sample));
                right[sample] = juce::jmap (fxMix, right[sample], wetBuffer.getSample (1, sample));
            }
        }
        else
        {
            for (int sample = 0; sample < numSamples; ++sample)
            {
                const auto dryL = left[sample];
                const auto dryR = right[sample];
                float wetL = dryL;
                float wetR = dryR;

                switch (renderParams.fxType)
                {
                    case 1:
                    {
                        const auto drive = 1.2f + renderParams.fxIntensity * 10.0f;
                        const auto bias = fxSpread * 0.6f - 0.3f;
                        const auto shape = 0.35f + fxColour * 0.65f;
                        const auto drivenL = std::tanh ((dryL + bias) * drive);
                        const auto drivenR = std::tanh ((dryR + bias) * drive);
                        wetL = juce::jlimit (-1.2f, 1.2f, drivenL * (0.7f + shape * 0.45f) + dryL * (0.4f - shape * 0.2f));
                        wetR = juce::jlimit (-1.2f, 1.2f, drivenR * (0.7f + shape * 0.45f) + dryR * (0.4f - shape * 0.2f));
                        break;
                    }
                    case 2:
                    {
                        const auto color = fxColour * 2.0f - 1.0f;
                        const auto bias = fxSpread * 0.9f - 0.45f;
                        const auto shapeDrive = 1.2f + renderParams.fxIntensity * 4.8f;
                        const auto charL = std::tanh ((dryL + bias) * shapeDrive) + color * dryL * std::abs (dryL) * 0.6f;
                        const auto charR = std::tanh ((dryR + bias) * shapeDrive) + color * dryR * std::abs (dryR) * 0.6f;
                        wetL = juce::jlimit (-1.2f, 1.2f, charL);
                        wetR = juce::jlimit (-1.2f, 1.2f, charR);
                        break;
                    }
                    case 6:
                    {
                        const auto modFreq = 8.0f + fxRate * 180.0f + fxColour * 520.0f;
                        const auto stereoPhase = 0.15f + fxSpread * 1.8f;
                        const auto phase = twoPi * (static_cast<float> (sample) / static_cast<float> (currentSampleRate)) * modFreq;
                        const auto mod = std::sin (phase);
                        wetL = dryL * mod;
                        wetR = dryR * std::cos (phase + stereoPhase);
                        break;
                    }
                    case 7:
                    {
                        const auto shiftHz = 10.0f + fxRate * 120.0f + fxColour * 1800.0f;
                        const auto stereoOffset = 1.0f + fxSpread * 0.18f;
                        const auto phase = twoPi * (static_cast<float> (sample) / static_cast<float> (currentSampleRate)) * shiftHz;
                        wetL = std::sin (std::asin (juce::jlimit (-0.99f, 0.99f, dryL)) + phase);
                        wetR = std::sin (std::asin (juce::jlimit (-0.99f, 0.99f, dryR)) + phase * stereoOffset);
                        break;
                    }
                    default:
                        break;
                }

                left[sample] = juce::jmap (fxMix, dryL, wetL);
                right[sample] = juce::jmap (fxMix, dryR, wetR);
            }
        }
    }

    for (int sample = 0; sample < numSamples; ++sample)
    {
        left[sample] = lowEqLeft.processSample (left[sample]);
        right[sample] = lowEqRight.processSample (right[sample]);
        left[sample] = midEqLeft.processSample (left[sample]);
        right[sample] = midEqRight.processSample (right[sample]);
        left[sample] = highEqLeft.processSample (left[sample]);
        right[sample] = highEqRight.processSample (right[sample]);
    }

    if (delaySend > 0.0001f && delayBuffer.getNumSamples() > 0)
    {
        const auto delaySamples = juce::jlimit (1, delayBuffer.getNumSamples() - 1,
                                                juce::roundToInt (renderParams.delayTimeSec * static_cast<float> (currentSampleRate)));
        const auto bufferSize = delayBuffer.getNumSamples();
        auto* delayLeft = delayBuffer.getWritePointer (0);
        auto* delayRight = delayBuffer.getWritePointer (1);

        for (int sample = 0; sample < numSamples; ++sample)
        {
            const auto readPosition = (delayWritePosition + bufferSize - delaySamples) % bufferSize;
            const auto wetL = delayLeft[readPosition];
            const auto wetR = delayRight[readPosition];
            const auto inputL = left[sample];
            const auto inputR = right[sample];

            delayLeft[delayWritePosition] = inputL * delaySend + wetL * feedback;
            delayRight[delayWritePosition] = inputR * delaySend + wetR * feedback;

            left[sample] = inputL + wetL;
            right[sample] = inputR + wetR;
            delayWritePosition = (delayWritePosition + 1) % bufferSize;
        }
    }

    if (reverbMix > 0.0001f)
    {
        juce::AudioBuffer<float> wetBuffer;
        wetBuffer.makeCopyOf (buffer, true);
        reverb.processStereo (wetBuffer.getWritePointer (0), wetBuffer.getWritePointer (1), wetBuffer.getNumSamples());

        buffer.applyGain (1.0f - reverbMix);
        buffer.addFrom (0, 0, wetBuffer, 0, 0, wetBuffer.getNumSamples(), reverbMix);
        buffer.addFrom (1, 0, wetBuffer, 1, 0, wetBuffer.getNumSamples(), reverbMix);
    }
}

void AdvancedVSTiAudioProcessor::applyEnvelopeSettings()
{
    for (auto& voice : voices)
    {
        voice.ampEnv.setParameters (renderParams.ampEnv);
        voice.filterEnv.setParameters (renderParams.filterEnv);
    }
}

juce::AudioProcessorEditor* AdvancedVSTiAudioProcessor::createEditor()
{
    return new AdvancedVSTiAudioProcessorEditor (*this);
}

void AdvancedVSTiAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    state.setProperty ("STATE_VERSION", pluginStateVersion, nullptr);
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AdvancedVSTiAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName (apvts.state.getType()))
    {
        auto restoredState = juce::ValueTree::fromXml (*xmlState);
        if constexpr (isDrumFlavor())
        {
            const auto savedStateVersion = static_cast<int> (restoredState.getProperty ("STATE_VERSION", 1));
            if (savedStateVersion < pluginStateVersion)
                migrateLegacyDrumFilterChoice (restoredState);
        }

        const juce::ScopedValueSetter<bool> suppress (suppressPresetCallback, true);
        apvts.replaceState (restoredState);
    }
    pendingPresetIndex.store (-1);
    currentProgramIndex = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), paramIndex (apvts, "PRESET"));
    updateRenderParameters();
    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
        refreshSampleBank();
    else
        refreshExternalPadSamples();
    applyEnvelopeSettings();
}

juce::AudioProcessorValueTreeState::ParameterLayout AdvancedVSTiAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    int oscDefault = 0;
    int polyphonyDefault = voiceLimitForFlavor();
    int unisonDefault = 1;
    float detuneDefault = 0.1f;
    float fmDefault = 0.0f;
    float syncDefault = 0.0f;
    float gateDefault = 8.0f;
    float osc1SemitoneDefault = 0.0f;
    float osc1DetuneDefault = 0.0f;
    float osc1PulseWidthDefault = 0.5f;
    int osc2TypeDefault = 1;
    float osc2SemitoneDefault = 0.0f;
    float osc2DetuneDefault = 0.02f;
    float osc2PulseWidthDefault = 0.5f;
    int osc3TypeDefault = static_cast<int> (OscType::square);
    float osc3SemitoneDefault = -12.0f;
    float osc3DetuneDefault = 0.0f;
    float osc3PulseWidthDefault = 0.5f;
    float osc2MixDefault = 0.35f;
    float subOscLevelDefault = 0.0f;
    float noiseLevelDefault = 0.0f;
    float ringModDefault = 0.0f;

    float ampAttackDefault = 0.01f;
    float ampDecayDefault = 0.2f;
    float ampSustainDefault = 0.8f;
    float ampReleaseDefault = 0.4f;

    float filtAttackDefault = 0.02f;
    float filtDecayDefault = 0.3f;
    float filtSustainDefault = 0.6f;
    float filtReleaseDefault = 0.4f;
    float envCurveDefault = 0.0f;

    int filterTypeDefault = 0;
    int filterSlopeDefault = 0;
    int filter2TypeDefault = 0;
    int filter2SlopeDefault = 0;
    float masterLevelDefault = 1.0f;
    float cutoffDefault = 1200.0f;
    float cutoff2Default = 2400.0f;
    float resonanceDefault = 0.4f;
    float resonance2Default = 0.4f;
    float resonanceMax = 10.0f;
    float filterEnvAmountDefault = 0.5f;
    float filterBalanceDefault = 0.0f;
    float panoramaDefault = 0.0f;
    float keyFollowDefault = 0.0f;
    int sampleBankDefault = 0;
    float sampleStartDefault = 0.0f;
    float sampleEndDefault = 1.0f;

    float lfo1RateDefault = 2.0f;
    int lfo1ShapeDefault = 0;
    bool lfo1EnvModeDefault = false;
    float lfo1AmountDefault = 0.0f;
    int lfo1DestinationDefault = 0;
    int lfo1AssignDestinationDefault = static_cast<int> (AdvancedVSTiAudioProcessor::MatrixDestination::off);
    float lfo1PitchDefault = 0.0f;

    float lfo2RateDefault = 3.0f;
    int lfo2ShapeDefault = 0;
    bool lfo2EnvModeDefault = false;
    float lfo2AmountDefault = 0.0f;
    int lfo2DestinationDefault = 5;
    int lfo2AssignDestinationDefault = static_cast<int> (AdvancedVSTiAudioProcessor::MatrixDestination::off);
    int lfo3ShapeDefault = 0;
    bool lfo3EnvModeDefault = true;
    float lfo3AmountDefault = 0.0f;
    int lfo3DestinationDefault = 10;
    int lfo3AssignDestinationDefault = static_cast<int> (AdvancedVSTiAudioProcessor::MatrixDestination::ampLevel);
    float lfo2FilterDefault = 0.0f;
    int fxTypeDefault = 0;
    float fxMixDefault = 0.0f;
    float fxIntensityDefault = 0.0f;
    float fxRateDefault = 0.65f;
    float fxColourDefault = 0.5f;
    float fxSpreadDefault = 0.4f;
    float delaySendDefault = 0.0f;
    float delayTimeDefault = 0.34f;
    float delayFeedbackDefault = 0.25f;
    float reverbMixDefault = 0.0f;
    float reverbTimeDefault = 0.45f;
    float reverbDampingDefault = 0.45f;
    float lowEqGainDefault = 0.0f;
    float lowEqFreqDefault = 220.0f;
    float lowEqQDefault = 0.8f;
    float midEqGainDefault = 0.0f;
    float midEqFreqDefault = 1400.0f;
    float midEqQDefault = 1.1f;
    float highEqGainDefault = 0.0f;
    float highEqFreqDefault = 5000.0f;
    float highEqQDefault = 0.8f;
    int saturationTypeDefault = 0;
    float drumMasterLevelDefault = 1.0f;
    float drumKickAttackDefault = 0.5f;
    float drumSnareToneDefault = 0.5f;
    float drumSnareSnappyDefault = 0.5f;

    int presetDefault = 0;
    int arpModeDefault = 0;
    float arpRateDefault = 4.0f;
    float rhythmRateDefault = 8.0f;
    float rhythmDepthDefault = 0.0f;
    std::array<float, 7> drumVoiceTuneDefaults {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    std::array<float, 6> drumVoiceDecayDefaults {
        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f
    };
    std::array<float, 15> drumVoiceLevelDefaults {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };

    if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
    {
        oscDefault = 3;
        detuneDefault = 0.0f;
        filterTypeDefault = 1;
        gateDefault = 0.42f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.18f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.08f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.09f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.08f;
        fmDefault = 150.0f;
        syncDefault = 0.28f;
        cutoffDefault = 14000.0f;
        resonanceDefault = 0.24f;
        filterEnvAmountDefault = 0.12f;
        rhythmRateDefault = 16.0f;
        rhythmDepthDefault = 0.0f;
        drumMasterLevelDefault = 1.0f;
        drumKickAttackDefault = 0.72f;
        drumSnareToneDefault = 0.56f;
        drumSnareSnappyDefault = 0.62f;
        drumVoiceTuneDefaults = drumMachineVoiceTuneDefaults;
        drumVoiceDecayDefaults = drumMachineVoiceDecayDefaults;
        drumVoiceLevelDefaults = drumMachineVoiceLevelDefaults;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        oscDefault = 0;
        detuneDefault = 0.0f;
        filterTypeDefault = 1;
        gateDefault = 0.95f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.34f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.16f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.1f;
        fmDefault = 90.0f;
        syncDefault = 0.08f;
        cutoffDefault = 9200.0f;
        resonanceDefault = 0.16f;
        filterEnvAmountDefault = 0.05f;
        rhythmRateDefault = 16.0f;
        rhythmDepthDefault = 0.0f;
        drumMasterLevelDefault = 1.0f;
        drumKickAttackDefault = 0.34f;
        drumSnareToneDefault = 0.48f;
        drumSnareSnappyDefault = 0.54f;
        drumVoiceTuneDefaults = drum808VoiceTuneDefaults;
        drumVoiceDecayDefaults = drum808VoiceDecayDefaults;
        drumVoiceLevelDefaults = drum808VoiceLevelDefaults;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        oscDefault = 4;
        detuneDefault = 0.0f;
        filterTypeDefault = 0;
        gateDefault = 1.0f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.2f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.05f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.1f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.05f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        cutoffDefault = 20000.0f;
        resonanceDefault = 0.1f;
        filterEnvAmountDefault = 0.0f;
        rhythmRateDefault = 16.0f;
        rhythmDepthDefault = 0.0f;
        drumMasterLevelDefault = 1.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        oscDefault = 1;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 0.0f;
        ampAttackDefault = 0.005f;
        ampDecayDefault = 0.16f;
        ampSustainDefault = 0.72f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.18f;
        filtSustainDefault = 0.22f;
        filtReleaseDefault = 0.12f;
        envCurveDefault = 0.06f;
        cutoffDefault = 240.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.36f;
        lfo1RateDefault = 0.1f;
        lfo2RateDefault = 0.2f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        oscDefault = 1;
        polyphonyDefault = 16;
        unisonDefault = 2;
        detuneDefault = 0.035f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.3f;
        ampDecayDefault = 0.7f;
        ampSustainDefault = 0.88f;
        ampReleaseDefault = 1.1f;
        filtAttackDefault = 0.12f;
        filtDecayDefault = 0.52f;
        filtSustainDefault = 0.72f;
        filtReleaseDefault = 0.9f;
        envCurveDefault = -0.08f;
        cutoffDefault = 3200.0f;
        resonanceDefault = 0.12f;
        filterEnvAmountDefault = 0.08f;
        lfo1RateDefault = 0.08f;
        lfo1PitchDefault = 0.0f;
        lfo2RateDefault = 0.08f;
        lfo2FilterDefault = 0.0f;
        rhythmDepthDefault = 0.0f;
        fxTypeDefault = 3;
        fxMixDefault = 0.16f;
        fxIntensityDefault = 0.24f;
        fxRateDefault = 0.42f;
        fxColourDefault = 0.54f;
        fxSpreadDefault = 0.48f;
        delaySendDefault = 0.08f;
        delayTimeDefault = 0.34f;
        delayFeedbackDefault = 0.22f;
        reverbMixDefault = 0.14f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        oscDefault = 1;
        unisonDefault = 1;
        detuneDefault = 0.01f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        ampAttackDefault = 0.002f;
        ampDecayDefault = 0.12f;
        ampSustainDefault = 0.7f;
        ampReleaseDefault = 0.1f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.1f;
        filtSustainDefault = 0.34f;
        filtReleaseDefault = 0.08f;
        envCurveDefault = 0.12f;
        cutoffDefault = 2200.0f;
        resonanceDefault = 0.22f;
        filterEnvAmountDefault = 0.28f;
        lfo1RateDefault = 3.2f;
        lfo1PitchDefault = 0.0f;
        lfo2RateDefault = 0.25f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        oscDefault = 1;
        unisonDefault = 2;
        detuneDefault = 0.04f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.5f;
        ampDecayDefault = 0.95f;
        ampSustainDefault = 0.9f;
        ampReleaseDefault = 1.5f;
        filtAttackDefault = 0.18f;
        filtDecayDefault = 0.72f;
        filtSustainDefault = 0.8f;
        filtReleaseDefault = 1.0f;
        envCurveDefault = -0.16f;
        cutoffDefault = 4200.0f;
        resonanceDefault = 0.12f;
        filterEnvAmountDefault = 0.08f;
        lfo1RateDefault = 0.08f;
        lfo1PitchDefault = 0.0f;
        lfo2RateDefault = 0.08f;
        lfo2FilterDefault = 0.0f;
        rhythmDepthDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        oscDefault = 2;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        gateDefault = 0.5f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.16f;
        ampSustainDefault = 0.12f;
        ampReleaseDefault = 0.12f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.14f;
        filtSustainDefault = 0.12f;
        filtReleaseDefault = 0.12f;
        envCurveDefault = 0.18f;
        cutoffDefault = 1800.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.44f;
        lfo1RateDefault = 0.1f;
        lfo2RateDefault = 0.14f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
    {
        oscDefault = 4;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 0.0f;
        syncDefault = 0.0f;
        gateDefault = 8.0f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.55f;
        ampSustainDefault = 0.86f;
        ampReleaseDefault = 0.42f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.3f;
        filtSustainDefault = 0.76f;
        filtReleaseDefault = 0.32f;
        envCurveDefault = 0.04f;
        cutoffDefault = 6200.0f;
        resonanceDefault = 0.18f;
        filterEnvAmountDefault = 0.1f;
        sampleBankDefault = 0;
        sampleStartDefault = 0.0f;
        sampleEndDefault = 0.92f;
        lfo1RateDefault = 0.12f;
        lfo2RateDefault = 0.1f;
        lfo2FilterDefault = 0.0f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
    {
        oscDefault = 1;
        unisonDefault = 1;
        detuneDefault = 0.0f;
        fmDefault = 460.0f;
        syncDefault = 0.08f;
        gateDefault = 0.55f;
        ampAttackDefault = 0.001f;
        ampDecayDefault = 0.32f;
        ampSustainDefault = 0.0f;
        ampReleaseDefault = 0.14f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.36f;
        filtSustainDefault = 0.0f;
        filtReleaseDefault = 0.12f;
        envCurveDefault = 0.45f;
        cutoffDefault = 540.0f;
        resonanceDefault = 0.82f;
        resonanceMax = 10.0f;
        filterEnvAmountDefault = 0.94f;
        lfo1RateDefault = 0.14f;
        lfo2RateDefault = 0.22f;
        lfo2FilterDefault = 0.06f;
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        oscDefault = 1;
        osc2TypeDefault = 2;
        unisonDefault = 2;
        masterLevelDefault = 1.0f;
        detuneDefault = 0.045f;
        osc2SemitoneDefault = 0.0f;
        osc2DetuneDefault = 0.08f;
        osc2MixDefault = 0.42f;
        subOscLevelDefault = 0.18f;
        noiseLevelDefault = 0.04f;
        ringModDefault = 0.08f;
        fmDefault = 120.0f;
        syncDefault = 0.24f;
        gateDefault = 3.2f;
        ampAttackDefault = 0.004f;
        ampDecayDefault = 0.28f;
        ampSustainDefault = 0.74f;
        ampReleaseDefault = 0.36f;
        filtAttackDefault = 0.001f;
        filtDecayDefault = 0.24f;
        filtSustainDefault = 0.42f;
        filtReleaseDefault = 0.24f;
        envCurveDefault = 0.08f;
        filterTypeDefault = 0;
        filter2TypeDefault = 2;
        cutoffDefault = 2200.0f;
        cutoff2Default = 5400.0f;
        resonanceDefault = 0.36f;
        resonance2Default = 0.24f;
        filterEnvAmountDefault = 0.42f;
        filterBalanceDefault = 0.28f;
        panoramaDefault = 0.0f;
        keyFollowDefault = 0.22f;
        lfo1RateDefault = 0.18f;
        lfo1PitchDefault = 0.0f;
        lfo1AmountDefault = 0.0f;
        lfo1DestinationDefault = 0;
        lfo2RateDefault = 0.22f;
        lfo2FilterDefault = 0.12f;
        lfo2AmountDefault = 0.12f;
        lfo2DestinationDefault = 5;
        lfo3AmountDefault = rhythmDepthDefault;
        lfo3DestinationDefault = 10;
        lfo3AssignDestinationDefault = static_cast<int> (AdvancedVSTiAudioProcessor::MatrixDestination::ampLevel);
        fxTypeDefault = 3;
        fxMixDefault = 0.18f;
        fxIntensityDefault = 0.32f;
        delaySendDefault = 0.16f;
        delayTimeDefault = 0.36f;
        delayFeedbackDefault = 0.28f;
        reverbMixDefault = 0.12f;
        reverbTimeDefault = 0.48f;
        reverbDampingDefault = 0.4f;
        lowEqGainDefault = 0.0f;
        midEqGainDefault = 0.0f;
        highEqGainDefault = 0.0f;
        saturationTypeDefault = 0;
    }

    const auto oscChoices = buildFlavor() == InstrumentFlavor::advanced ? advancedOscillatorChoices() : classicOscillatorChoices();

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("PRESET", "Preset", presetChoicesForFlavor(), presetDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSCTYPE", "Osc Type", oscChoices, oscDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSC2TYPE", "Osc 2 Type", oscChoices, osc2TypeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("OSC3TYPE", "Osc 3 Type", oscChoices, osc3TypeDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("SAMPLEBANK", "Sample Bank", sampleBankChoices(), sampleBankDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SAMPLESTART", "Sample Start", 0.0f, 0.95f, sampleStartDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SAMPLEEND", "Sample End", 0.05f, 1.0f, sampleEndDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("MASTERLEVEL", "Master Level", 0.0f, 1.5f, masterLevelDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
        params.push_back (std::make_unique<juce::AudioParameterInt> ("POLYPHONY", "Polyphony", 1, voiceLimitForFlavor(), polyphonyDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("MONOENABLE", "Mono Enabled", false));
    params.push_back (std::make_unique<juce::AudioParameterInt> ("UNISON", "Unison", 1, maxUnisonForFlavor(), unisonDefault));
    if constexpr (isSynthDrumFlavor())
    {
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DETUNE", "Detune", -6.0f, 6.0f, detuneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("FMAMOUNT", "FM Amount", 0.0f, 200.0f, fmDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("SYNC", "Sync", 0.0f, 1.0f, syncDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSCGATE", "Osc Note Length Gate", 0.1f, 2.4f, gateDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DRUMMASTERLEVEL", "Drum Master Level", 0.0f, 1.5f, drumMasterLevelDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DRUM_KICK_ATTACK", "Bass Drum Attack", 0.0f, 1.0f, drumKickAttackDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DRUM_SNARE_TONE", "Snare Tone", 0.0f, 1.0f, drumSnareToneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DRUM_SNARE_SNAPPY", "Snare Snappy", 0.0f, 1.0f, drumSnareSnappyDefault));

        for (size_t index = 0; index < drumVoiceTuneParamIds.size(); ++index)
        {
            params.push_back (std::make_unique<juce::AudioParameterFloat> (drumVoiceTuneParamIds[index],
                                                                            drumVoiceTuneNames[index],
                                                                            -12.0f, 12.0f,
                                                                            drumVoiceTuneDefaults[index]));
        }

        for (size_t index = 0; index < drumVoiceDecayParamIds.size(); ++index)
        {
            params.push_back (std::make_unique<juce::AudioParameterFloat> (drumVoiceDecayParamIds[index],
                                                                            drumVoiceDecayNames[index],
                                                                            0.0f, 1.0f,
                                                                            drumVoiceDecayDefaults[index]));
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DETUNE", "Detune", 0.0f, 1.0f, 0.0f));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("FMAMOUNT", "FM Amount", 0.0f, 1.0f, 0.0f));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("SYNC", "Sync", 0.0f, 1.0f, 0.0f));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSCGATE", "Osc Note Length Gate", 0.1f, 2.4f, gateDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DRUMMASTERLEVEL", "Drum Master Level", 0.0f, 1.5f, drumMasterLevelDefault));
    }
    else
    {
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("DETUNE", "Detune", 0.0f, 1.0f, detuneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("FMAMOUNT", "FM Amount", 0.0f, 1000.0f, fmDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("SYNC", "Sync", 0.0f, 4.0f, syncDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSCGATE", "Osc Note Length Gate", 0.01f, 8.0f, gateDefault));
    }
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC1SEMITONE", "Osc 1 Semitone", -24.0f, 24.0f, osc1SemitoneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC1DETUNE", "Osc 1 Detune", -1.0f, 1.0f, osc1DetuneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC1PW", "Osc 1 Pulse Width", 0.05f, 0.95f, osc1PulseWidthDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC2PW", "Osc 2 Pulse Width", 0.05f, 0.95f, osc2PulseWidthDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC3SEMITONE", "Osc 3 Semitone", -24.0f, 24.0f, osc3SemitoneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC3DETUNE", "Osc 3 Detune", -1.0f, 1.0f, osc3DetuneDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC3PW", "Osc 3 Pulse Width", 0.05f, 0.95f, osc3PulseWidthDefault));
    }
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC2SEMITONE", "Osc 2 Semitone", -24.0f, 24.0f, osc2SemitoneDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC2DETUNE", "Osc 2 Detune", -1.0f, 1.0f, osc2DetuneDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("OSC2MIX", "Osc 2 Mix", 0.0f, 1.0f, osc2MixDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("SUBOSCLEVEL", "Sub Osc Level", 0.0f, 1.0f, subOscLevelDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("OSC3ENABLE", "Osc 3 Enabled", subOscLevelDefault > 0.0001f));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("NOISELEVEL", "Noise Level", 0.0f, 1.0f, noiseLevelDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RINGMOD", "Ring Mod", 0.0f, 1.0f, ringModDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        params.push_back (std::make_unique<juce::AudioParameterBool> ("RINGMODENABLE", "Ring Mod Enabled", true));
        params.push_back (std::make_unique<juce::AudioParameterBool> ("FMENABLE", "FM Enabled", true));
        params.push_back (std::make_unique<juce::AudioParameterBool> ("SYNCENABLE", "Sync Enabled", true));
    }

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPATTACK", "Amp Attack", 0.001f, 10.0f, ampAttackDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPDECAY", "Amp Decay", 0.001f, 10.0f, ampDecayDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPSUSTAIN", "Amp Sustain", 0.0f, 1.0f, ampSustainDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("AMPRELEASE", "Amp Release", 0.001f, 10.0f, ampReleaseDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTATTACK", "Filter Attack", 0.001f, 10.0f, filtAttackDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTDECAY", "Filter Decay", 0.001f, 10.0f, filtDecayDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTSUSTAIN", "Filter Sustain", 0.0f, 1.0f, filtSustainDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTRELEASE", "Filter Release", 0.001f, 10.0f, filtReleaseDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("ENVCURVE", "Envelope Curve", -1.0f, 1.0f, envCurveDefault));

    if constexpr (isDrumFlavor())
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTERTYPE", "Filter Type", juce::StringArray { "Off", "LP", "BP", "HP", "Notch" }, filterTypeDefault));
    else
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTERTYPE", "Filter Type", juce::StringArray { "LP", "BP", "HP", "Notch" }, filterTypeDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTER2TYPE", "Filter 2 Type", juce::StringArray { "LP", "BP", "HP", "Notch" }, filter2TypeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        params.push_back (std::make_unique<juce::AudioParameterBool> ("FILTER1ENABLE", "Filter 1 Enabled", true));
        params.push_back (std::make_unique<juce::AudioParameterBool> ("FILTER2ENABLE", "Filter 2 Enabled", true));
    }
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("CUTOFF", "Cutoff", 20.0f, 20000.0f, cutoffDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("CUTOFF2", "Cutoff 2", 20.0f, 20000.0f, cutoff2Default));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RESONANCE", "Resonance", 0.1f, resonanceMax, resonanceDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("RESONANCE2", "Resonance 2", 0.1f, resonanceMax, resonance2Default));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTERENVAMOUNT", "Filter Env Amount", 0.0f, 1.0f, filterEnvAmountDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FILTERBALANCE", "Filter Balance", 0.0f, 1.0f, filterBalanceDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("PANORAMA", "Panorama", -1.0f, 1.0f, panoramaDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("KEYFOLLOW", "Keyfollow", 0.0f, 1.0f, keyFollowDefault));
    }

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1RATE", "LFO1 Rate", 0.05f, 40.0f, lfo1RateDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO1SHAPE", "LFO1 Shape", juce::StringArray { "Sine", "Triangle", "Saw", "Noise", "Square" }, lfo1ShapeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("LFO1ENABLE", "LFO1 Enabled", true));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("LFO1ENVMODE", "LFO1 Env Mode", lfo1EnvModeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1AMOUNT", "LFO1 Amount", -1.0f, 1.0f, lfo1AmountDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO1DEST", "LFO1 Destination", virusLfoDestinationChoices(), lfo1DestinationDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO1ASSIGNDEST", "LFO1 Assign Destination", matrixDestinationChoices(), lfo1AssignDestinationDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO1PITCH", "LFO1 -> Pitch", 0.0f, 24.0f, lfo1PitchDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2RATE", "LFO2 Rate", 0.05f, 40.0f, lfo2RateDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO2SHAPE", "LFO2 Shape", juce::StringArray { "Sine", "Triangle", "Saw", "Noise", "Square" }, lfo2ShapeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("LFO2ENABLE", "LFO2 Enabled", true));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("LFO2ENVMODE", "LFO2 Env Mode", lfo2EnvModeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2AMOUNT", "LFO2 Amount", -1.0f, 1.0f, lfo2AmountDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO2DEST", "LFO2 Destination", virusLfoDestinationChoices(), lfo2DestinationDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO2ASSIGNDEST", "LFO2 Assign Destination", matrixDestinationChoices(), lfo2AssignDestinationDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO2FILTER", "LFO2 -> Filter", 0.0f, 1.0f, lfo2FilterDefault));

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("ARPMODE", "Arp Mode", juce::StringArray { "Up", "Down", "UpDown", "Random" }, arpModeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("ARPENABLE", "Arp Enabled", false));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("ARPRATE", "Arp Rate", 0.25f, 16.0f, arpRateDefault));

    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RHYTHMGATE_RATE", "Rhythm Gate Rate", 0.25f, 32.0f, rhythmRateDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("LFO3ENABLE", "LFO3 Enabled", true));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO3SHAPE", "LFO3 Shape", juce::StringArray { "Sine", "Triangle", "Saw", "Noise", "Square" }, lfo3ShapeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterBool> ("LFO3ENVMODE", "LFO3 Env Mode", lfo3EnvModeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("LFO3AMOUNT", "LFO3 Amount", 0.0f, 1.0f, lfo3AmountDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO3DEST", "LFO3 Destination", virusLfoDestinationChoices(), lfo3DestinationDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("LFO3ASSIGNDEST", "LFO3 Assign Destination", matrixDestinationChoices(), lfo3AssignDestinationDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("RHYTHMGATE_DEPTH", "Rhythm Gate Depth", 0.0f, 1.0f, rhythmDepthDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("FXTYPE", "FX Type", juce::StringArray { "Off", "Distortion", "Character", "Chorus", "Phaser", "Flanger", "Ring Mod", "Freq Shift" }, fxTypeDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterChoice> ("SATURATIONTYPE", "Saturation Type", juce::StringArray { "Soft", "Warm", "Drive", "Fold" }, saturationTypeDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FXMIX", "FX Mix", 0.0f, 1.0f, fxMixDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("FXINTENSITY", "FX Intensity", 0.0f, 1.0f, fxIntensityDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("FXRATE", "FX Rate", 0.02f, 12.0f, fxRateDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("FXCOLOUR", "FX Colour", 0.0f, 1.0f, fxColourDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("FXSPREAD", "FX Spread", 0.0f, 1.0f, fxSpreadDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("DELAYSEND", "Delay Send", 0.0f, 1.0f, delaySendDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("DELAYTIME", "Delay Time", 0.05f, 1.2f, delayTimeDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("DELAYFEEDBACK", "Delay Feedback", 0.0f, 0.92f, delayFeedbackDefault));
    params.push_back (std::make_unique<juce::AudioParameterFloat> ("REVERBMIX", "Reverb Mix", 0.0f, 1.0f, reverbMixDefault));
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("REVERBTIME", "Reverb Time", 0.1f, 1.0f, reverbTimeDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("REVERBDAMP", "Reverb Damping", 0.0f, 1.0f, reverbDampingDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("LOWEQGAIN", "Low EQ Gain", -16.0f, 16.0f, lowEqGainDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("LOWEQFREQ", "Low EQ Freq", 40.0f, 1200.0f, lowEqFreqDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("LOWEQQ", "Low EQ Q", 0.3f, 2.5f, lowEqQDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("MIDEQGAIN", "Mid EQ Gain", -16.0f, 16.0f, midEqGainDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("MIDEQFREQ", "Mid EQ Freq", 120.0f, 8000.0f, midEqFreqDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("MIDEQQ", "Mid EQ Q", 0.3f, 8.0f, midEqQDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("HIGHEQGAIN", "High EQ Gain", -16.0f, 16.0f, highEqGainDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("HIGHEQFREQ", "High EQ Freq", 1200.0f, 16000.0f, highEqFreqDefault));
        params.push_back (std::make_unique<juce::AudioParameterFloat> ("HIGHEQQ", "High EQ Q", 0.3f, 2.5f, highEqQDefault));

        for (int slotIndex = 0; slotIndex < 6; ++slotIndex)
        {
            const auto slotLabel = juce::String ("Matrix ") + juce::String (slotIndex + 1);
            params.push_back (std::make_unique<juce::AudioParameterChoice> ("MATRIX" + juce::String (slotIndex + 1) + "SOURCE",
                                                                            slotLabel + " Source",
                                                                            matrixSourceChoices(),
                                                                            0));
            for (int targetIndex = 0; targetIndex < 3; ++targetIndex)
            {
                const auto targetLabel = juce::String (" Target ") + juce::String (targetIndex + 1);
                params.push_back (std::make_unique<juce::AudioParameterFloat> ("MATRIX" + juce::String (slotIndex + 1) + "AMOUNT" + juce::String (targetIndex + 1),
                                                                                slotLabel + targetLabel + " Amount",
                                                                                -1.0f,
                                                                                1.0f,
                                                                                0.0f));
                params.push_back (std::make_unique<juce::AudioParameterChoice> ("MATRIX" + juce::String (slotIndex + 1) + "DEST" + juce::String (targetIndex + 1),
                                                                                slotLabel + targetLabel + " Destination",
                                                                                matrixDestinationChoices(),
                                                                                0));
            }
        }
    }

    if constexpr (isSynthDrumFlavor())
    {
        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
        {
            params.push_back (std::make_unique<juce::AudioParameterFloat> (drumVoiceLevelParamIds[index],
                                                                            drumVoiceLevelNames[index],
                                                                            0.0f, 1.5f,
                                                                            drumVoiceLevelDefaults[index]));
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        const auto sustainRange = juce::NormalisableRange<float> (0.0f, 120.0f, 0.01f, 0.35f);
        const auto releaseRange = juce::NormalisableRange<float> (0.0f, 12.0f, 0.01f, 0.4f);
        for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
        {
            const auto padTitle = juce::String (vecPadTitles[static_cast<size_t> (padIndex)]);
            params.push_back (std::make_unique<juce::AudioParameterFloat> (externalPadLevelParameterIdForIndex (padIndex),
                                                                            padTitle + " Level",
                                                                            0.0f, 1.5f,
                                                                            1.0f));
            params.push_back (std::make_unique<juce::AudioParameterFloat> (externalPadSustainParameterIdForIndex (padIndex),
                                                                            padTitle + " Sustain",
                                                                            sustainRange,
                                                                            120.0f));
            params.push_back (std::make_unique<juce::AudioParameterFloat> (externalPadReleaseParameterIdForIndex (padIndex),
                                                                            padTitle + " Release",
                                                                            releaseRange,
                                                                            0.2f));
            params.push_back (std::make_unique<juce::AudioParameterInt> (externalPadSampleParameterIdForIndex (padIndex),
                                                                          padTitle + " Sample",
                                                                          0,
                                                                          maxVecPadSampleChoices - 1,
                                                                          0));
        }
    }

    params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTERSLOPE", "Filter Slope",
                                                                    juce::StringArray { "12 dB", "16 dB", "24 dB" },
                                                                    filterSlopeDefault));
    params.push_back (std::make_unique<juce::AudioParameterChoice> ("FILTER2SLOPE", "Filter 2 Slope",
                                                                    juce::StringArray { "12 dB", "16 dB", "24 dB" },
                                                                    filter2SlopeDefault));

    return { params.begin(), params.end() };
}

void AdvancedVSTiAudioProcessor::setParameterActual (const char* paramId, float value)
{
    auto* parameter = apvts.getParameter (paramId);
    if (parameter == nullptr)
        return;

    parameter->beginChangeGesture();
    parameter->setValueNotifyingHost (parameter->convertTo0to1 (value));
    parameter->endChangeGesture();
}

void AdvancedVSTiAudioProcessor::applyPresetByIndex (int presetIndex)
{
    const juce::ScopedValueSetter<bool> suppress (suppressPresetCallback, true);
    currentProgramIndex = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), presetIndex);

    setParameterActual ("LFO1PITCH", 0.0f);
    setParameterActual ("LFO2FILTER", 0.0f);
    setParameterActual ("RHYTHMGATE_DEPTH", 0.0f);
    setParameterActual ("ARPMODE", 0.0f);
    setParameterActual ("ARPRATE", 4.0f);
    setParameterActual ("LFO1RATE", 0.1f);
    setParameterActual ("LFO2RATE", 0.1f);
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        setParameterActual ("LFO1ENVMODE", 0.0f);
        setParameterActual ("LFO2ENVMODE", 0.0f);
        setParameterActual ("LFO3SHAPE", 0.0f);
        setParameterActual ("LFO3ENVMODE", 1.0f);
    }
    setParameterActual ("FILTERSLOPE", 0.0f);
    setParameterActual ("FILTER2SLOPE", 0.0f);
    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        setParameterActual ("MASTERLEVEL", 1.0f);
        setParameterActual ("MONOENABLE", 0.0f);
        setParameterActual ("ARPENABLE", 0.0f);
        setParameterActual ("LFO1ENABLE", 1.0f);
        setParameterActual ("LFO2ENABLE", 1.0f);
        setParameterActual ("LFO3ENABLE", 1.0f);
        setParameterActual ("PANORAMA", 0.0f);
        setParameterActual ("KEYFOLLOW", 0.22f);
        setParameterActual ("RESONANCE2", 0.24f);
        setParameterActual ("SATURATIONTYPE", 0.0f);
        setParameterActual ("FXTYPE", 0.0f);
        setParameterActual ("FXMIX", 0.16f);
        setParameterActual ("FXINTENSITY", 0.24f);
        setParameterActual ("FXRATE", 0.42f);
        setParameterActual ("FXCOLOUR", 0.54f);
        setParameterActual ("FXSPREAD", 0.48f);
        setParameterActual ("DELAYSEND", 0.08f);
        setParameterActual ("DELAYTIME", 0.34f);
        setParameterActual ("DELAYFEEDBACK", 0.22f);
        setParameterActual ("REVERBMIX", 0.12f);
        setParameterActual ("REVERBTIME", 0.48f);
        setParameterActual ("REVERBDAMP", 0.4f);
        setParameterActual ("LOWEQGAIN", 0.0f);
        setParameterActual ("LOWEQFREQ", 220.0f);
        setParameterActual ("LOWEQQ", 0.8f);
        setParameterActual ("MIDEQGAIN", 0.0f);
        setParameterActual ("MIDEQFREQ", 1400.0f);
        setParameterActual ("MIDEQQ", 1.1f);
        setParameterActual ("HIGHEQGAIN", 0.0f);
        setParameterActual ("HIGHEQFREQ", 5000.0f);
        setParameterActual ("HIGHEQQ", 0.8f);
    }

    if constexpr (buildFlavor() == InstrumentFlavor::bassSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 320.0f);
                setParameterActual ("RESONANCE", 0.2f);
                setParameterActual ("FILTERENVAMOUNT", 0.34f);
                setParameterActual ("AMPDECAY", 0.16f);
                setParameterActual ("AMPSUSTAIN", 0.76f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 210.0f);
                setParameterActual ("RESONANCE", 0.16f);
                setParameterActual ("FILTERENVAMOUNT", 0.28f);
                setParameterActual ("AMPDECAY", 0.14f);
                setParameterActual ("AMPSUSTAIN", 0.7f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 430.0f);
                setParameterActual ("RESONANCE", 0.22f);
                setParameterActual ("FILTERENVAMOUNT", 0.46f);
                setParameterActual ("AMPDECAY", 0.11f);
                setParameterActual ("AMPSUSTAIN", 0.42f);
                setParameterActual ("AMPRELEASE", 0.09f);
                break;
            default:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("CUTOFF", 170.0f);
                setParameterActual ("RESONANCE", 0.14f);
                setParameterActual ("FILTERENVAMOUNT", 0.2f);
                setParameterActual ("AMPDECAY", 0.14f);
                setParameterActual ("AMPSUSTAIN", 0.82f);
                setParameterActual ("AMPRELEASE", 0.1f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::stringSynth)
    {
        auto applyStringSettings = [this] (std::initializer_list<std::pair<const char*, float>> settings)
        {
            for (const auto& setting : settings)
                setParameterActual (setting.first, setting.second);
        };

        applyStringSettings ({
            { "POLYPHONY", 16.0f },
            { "OSCTYPE", 1.0f },
            { "UNISON", 4.0f },
            { "DETUNE", 0.035f },
            { "OSCGATE", 8.0f },
            { "FILTERTYPE", 0.0f },
            { "CUTOFF", 3200.0f },
            { "RESONANCE", 0.12f },
            { "FILTERENVAMOUNT", 0.08f },
            { "AMPATTACK", 0.3f },
            { "AMPDECAY", 0.7f },
            { "AMPSUSTAIN", 0.88f },
            { "AMPRELEASE", 1.1f },
            { "FILTATTACK", 0.12f },
            { "FILTDECAY", 0.52f },
            { "FILTSUSTAIN", 0.72f },
            { "FILTRELEASE", 0.9f },
            { "FXTYPE", 3.0f },
            { "FXMIX", 0.16f },
            { "FXINTENSITY", 0.24f },
            { "DELAYSEND", 0.08f },
            { "DELAYTIME", 0.34f },
            { "DELAYFEEDBACK", 0.22f },
            { "REVERBMIX", 0.14f },
        });

        switch (presetIndex)
        {
            case 1:
                applyStringSettings ({
                    { "OSCTYPE", 2.0f }, { "UNISON", 4.0f }, { "DETUNE", 0.028f }, { "CUTOFF", 2500.0f },
                    { "AMPATTACK", 0.48f }, { "AMPRELEASE", 1.5f }, { "FXMIX", 0.12f }, { "REVERBMIX", 0.22f },
                });
                break;
            case 2:
                applyStringSettings ({
                    { "UNISON", 2.0f }, { "DETUNE", 0.018f }, { "OSCGATE", 2.2f }, { "CUTOFF", 1800.0f },
                    { "RESONANCE", 0.18f }, { "FILTERENVAMOUNT", 0.24f }, { "AMPATTACK", 0.03f }, { "AMPDECAY", 0.26f },
                    { "AMPSUSTAIN", 0.66f }, { "AMPRELEASE", 0.28f }, { "FILTATTACK", 0.001f }, { "FILTDECAY", 0.18f },
                    { "FILTSUSTAIN", 0.16f }, { "FILTRELEASE", 0.1f }, { "FXTYPE", 1.0f }, { "FXMIX", 0.04f },
                    { "FXINTENSITY", 0.12f }, { "DELAYSEND", 0.02f }, { "REVERBMIX", 0.06f },
                });
                break;
            case 3:
                applyStringSettings ({
                    { "OSCTYPE", 2.0f }, { "POLYPHONY", 24.0f }, { "UNISON", 4.0f }, { "DETUNE", 0.034f },
                    { "CUTOFF", 2200.0f }, { "FILTERENVAMOUNT", 0.12f }, { "AMPATTACK", 0.36f }, { "AMPRELEASE", 1.9f },
                    { "FXMIX", 0.18f }, { "FXINTENSITY", 0.28f }, { "REVERBMIX", 0.28f },
                });
                break;
            case 4:
                applyStringSettings ({
                    { "POLYPHONY", 32.0f }, { "UNISON", 6.0f }, { "DETUNE", 0.06f }, { "CUTOFF", 4000.0f },
                    { "FILTERENVAMOUNT", 0.18f }, { "AMPATTACK", 1.2f }, { "AMPRELEASE", 2.8f }, { "FXMIX", 0.24f },
                    { "FXINTENSITY", 0.34f }, { "DELAYSEND", 0.14f }, { "DELAYTIME", 0.46f }, { "DELAYFEEDBACK", 0.32f },
                    { "REVERBMIX", 0.24f },
                });
                break;
            case 5:
                applyStringSettings ({
                    { "UNISON", 4.0f }, { "DETUNE", 0.032f }, { "CUTOFF", 3300.0f }, { "AMPATTACK", 0.22f },
                    { "AMPDECAY", 0.52f }, { "AMPSUSTAIN", 0.82f }, { "AMPRELEASE", 0.9f }, { "FILTATTACK", 0.08f },
                    { "FILTDECAY", 0.36f }, { "FILTSUSTAIN", 0.58f }, { "FILTRELEASE", 0.56f }, { "FXMIX", 0.08f },
                    { "DELAYSEND", 0.04f }, { "REVERBMIX", 0.1f },
                });
                break;
            case 6:
                applyStringSettings ({
                    { "UNISON", 6.0f }, { "DETUNE", 0.044f }, { "CUTOFF", 4100.0f }, { "AMPATTACK", 0.18f },
                    { "AMPRELEASE", 1.2f }, { "FXMIX", 0.24f }, { "FXINTENSITY", 0.36f }, { "DELAYSEND", 0.05f },
                    { "REVERBMIX", 0.18f },
                });
                break;
            case 7:
                applyStringSettings ({
                    { "OSCTYPE", 2.0f }, { "UNISON", 2.0f }, { "DETUNE", 0.015f }, { "OSCGATE", 3.4f }, { "CUTOFF", 2600.0f },
                    { "RESONANCE", 0.2f }, { "FILTERENVAMOUNT", 0.3f }, { "AMPATTACK", 0.08f }, { "AMPDECAY", 0.25f },
                    { "AMPSUSTAIN", 0.7f }, { "AMPRELEASE", 0.35f }, { "FILTATTACK", 0.001f }, { "FILTDECAY", 0.22f },
                    { "FILTSUSTAIN", 0.24f }, { "FILTRELEASE", 0.18f }, { "FXTYPE", 2.0f }, { "FXMIX", 0.14f },
                    { "FXINTENSITY", 0.22f },
                });
                break;
            case 8:
                applyStringSettings ({
                    { "POLYPHONY", 32.0f }, { "UNISON", 5.0f }, { "DETUNE", 0.052f }, { "CUTOFF", 3000.0f },
                    { "FILTERENVAMOUNT", 0.1f }, { "AMPATTACK", 0.9f }, { "AMPRELEASE", 2.4f }, { "FXMIX", 0.18f },
                    { "FXINTENSITY", 0.28f }, { "REVERBMIX", 0.24f },
                });
                break;
            case 9:
                applyStringSettings ({
                    { "POLYPHONY", 12.0f }, { "UNISON", 2.0f }, { "DETUNE", 0.012f }, { "OSCGATE", 2.0f }, { "CUTOFF", 1400.0f },
                    { "FILTERENVAMOUNT", 0.18f }, { "AMPATTACK", 0.02f }, { "AMPDECAY", 0.3f }, { "AMPSUSTAIN", 0.52f },
                    { "AMPRELEASE", 0.38f }, { "FILTATTACK", 0.001f }, { "FILTDECAY", 0.18f }, { "FILTSUSTAIN", 0.18f },
                    { "FILTRELEASE", 0.18f }, { "FXTYPE", 0.0f }, { "FXMIX", 0.0f }, { "DELAYSEND", 0.0f }, { "REVERBMIX", 0.08f },
                });
                break;
            case 10:
                applyStringSettings ({
                    { "POLYPHONY", 24.0f }, { "UNISON", 4.0f }, { "DETUNE", 0.03f }, { "CUTOFF", 3600.0f },
                    { "FILTERENVAMOUNT", 0.1f }, { "AMPATTACK", 0.24f }, { "AMPRELEASE", 1.2f }, { "FXTYPE", 2.0f },
                    { "FXMIX", 0.2f }, { "FXINTENSITY", 0.4f }, { "REVERBMIX", 0.18f },
                });
                break;
            case 11:
                applyStringSettings ({
                    { "POLYPHONY", 8.0f }, { "UNISON", 3.0f }, { "DETUNE", 0.02f }, { "CUTOFF", 1200.0f },
                    { "RESONANCE", 0.24f }, { "FILTERENVAMOUNT", 0.06f }, { "AMPATTACK", 0.16f }, { "AMPRELEASE", 1.4f },
                    { "FXTYPE", 0.0f }, { "FXMIX", 0.0f }, { "DELAYSEND", 0.0f }, { "REVERBMIX", 0.12f },
                });
                break;
            case 12:
                applyStringSettings ({
                    { "UNISON", 4.0f }, { "DETUNE", 0.04f }, { "CUTOFF", 5200.0f }, { "FILTERENVAMOUNT", 0.28f },
                    { "AMPATTACK", 0.35f }, { "AMPRELEASE", 1.6f }, { "FILTATTACK", 0.04f }, { "FILTDECAY", 0.7f },
                    { "FILTSUSTAIN", 0.48f }, { "FILTRELEASE", 1.1f }, { "FXTYPE", 4.0f }, { "FXMIX", 0.18f },
                    { "FXINTENSITY", 0.26f }, { "DELAYSEND", 0.08f }, { "REVERBMIX", 0.16f },
                });
                break;
            case 13:
                applyStringSettings ({
                    { "POLYPHONY", 32.0f }, { "UNISON", 6.0f }, { "DETUNE", 0.065f }, { "CUTOFF", 4600.0f },
                    { "AMPATTACK", 1.4f }, { "AMPRELEASE", 3.0f }, { "FXMIX", 0.28f }, { "FXINTENSITY", 0.36f },
                    { "DELAYSEND", 0.18f }, { "DELAYTIME", 0.52f }, { "DELAYFEEDBACK", 0.4f }, { "REVERBMIX", 0.3f },
                });
                break;
            case 14:
                applyStringSettings ({
                    { "POLYPHONY", 48.0f }, { "UNISON", 8.0f }, { "DETUNE", 0.08f }, { "CUTOFF", 3400.0f },
                    { "FILTERENVAMOUNT", 0.14f }, { "AMPATTACK", 0.8f }, { "AMPRELEASE", 2.5f }, { "FXMIX", 0.2f },
                    { "DELAYSEND", 0.1f }, { "REVERBMIX", 0.24f },
                });
                break;
            case 15:
                applyStringSettings ({
                    { "POLYPHONY", 4.0f }, { "UNISON", 2.0f }, { "DETUNE", 0.01f }, { "OSCGATE", 2.8f }, { "CUTOFF", 2100.0f },
                    { "RESONANCE", 0.22f }, { "FILTERENVAMOUNT", 0.32f }, { "AMPATTACK", 0.01f }, { "AMPDECAY", 0.18f },
                    { "AMPSUSTAIN", 0.78f }, { "AMPRELEASE", 0.2f }, { "FILTATTACK", 0.001f }, { "FILTDECAY", 0.18f },
                    { "FILTSUSTAIN", 0.2f }, { "FILTRELEASE", 0.1f }, { "FXTYPE", 1.0f }, { "FXMIX", 0.08f },
                    { "FXINTENSITY", 0.18f }, { "DELAYSEND", 0.02f }, { "REVERBMIX", 0.04f },
                });
                break;
            case 16:
                applyStringSettings ({
                    { "UNISON", 4.0f }, { "DETUNE", 0.045f }, { "CUTOFF", 1800.0f }, { "RESONANCE", 0.2f },
                    { "FILTERENVAMOUNT", 0.4f }, { "AMPATTACK", 0.12f }, { "AMPRELEASE", 1.8f }, { "FILTATTACK", 0.3f },
                    { "FILTDECAY", 1.6f }, { "FILTSUSTAIN", 0.12f }, { "FILTRELEASE", 1.2f }, { "FXTYPE", 2.0f },
                    { "FXMIX", 0.16f }, { "FXINTENSITY", 0.42f }, { "DELAYSEND", 0.1f }, { "REVERBMIX", 0.16f },
                });
                break;
            case 17:
                applyStringSettings ({
                    { "POLYPHONY", 24.0f }, { "UNISON", 5.0f }, { "DETUNE", 0.05f }, { "CUTOFF", 2400.0f },
                    { "FILTERENVAMOUNT", 0.14f }, { "AMPATTACK", 0.55f }, { "AMPRELEASE", 2.2f }, { "FXMIX", 0.2f },
                    { "FXINTENSITY", 0.3f }, { "DELAYSEND", 0.12f }, { "REVERBMIX", 0.34f },
                });
                break;
            case 18:
                applyStringSettings ({
                    { "POLYPHONY", 12.0f }, { "UNISON", 4.0f }, { "DETUNE", 0.022f }, { "CUTOFF", 900.0f },
                    { "RESONANCE", 0.3f }, { "FILTERENVAMOUNT", 0.05f }, { "AMPATTACK", 0.4f }, { "AMPDECAY", 0.8f },
                    { "AMPSUSTAIN", 0.7f }, { "AMPRELEASE", 1.5f }, { "FILTATTACK", 0.6f }, { "FILTDECAY", 1.8f },
                    { "FILTSUSTAIN", 0.4f }, { "FILTRELEASE", 1.4f }, { "FXTYPE", 4.0f }, { "FXMIX", 0.12f },
                    { "FXINTENSITY", 0.18f }, { "REVERBMIX", 0.26f },
                });
                break;
            case 19:
                applyStringSettings ({
                    { "UNISON", 4.0f }, { "DETUNE", 0.035f }, { "OSCGATE", 1.6f }, { "CUTOFF", 2000.0f },
                    { "FILTERENVAMOUNT", 0.36f }, { "AMPATTACK", 0.01f }, { "AMPDECAY", 0.22f }, { "AMPSUSTAIN", 0.46f },
                    { "AMPRELEASE", 0.18f }, { "FILTATTACK", 0.001f }, { "FILTDECAY", 0.16f }, { "FILTSUSTAIN", 0.14f },
                    { "FILTRELEASE", 0.14f }, { "FXMIX", 0.1f }, { "DELAYSEND", 0.06f }, { "REVERBMIX", 0.08f },
                });
                break;
            case 20:
                applyStringSettings ({
                    { "POLYPHONY", 32.0f }, { "UNISON", 8.0f }, { "DETUNE", 0.095f }, { "CUTOFF", 2800.0f },
                    { "AMPATTACK", 0.7f }, { "AMPRELEASE", 2.7f }, { "FXMIX", 0.3f }, { "FXINTENSITY", 0.42f },
                    { "DELAYSEND", 0.12f }, { "DELAYTIME", 0.38f }, { "DELAYFEEDBACK", 0.28f }, { "REVERBMIX", 0.25f },
                });
                break;
            case 21:
                applyStringSettings ({
                    { "UNISON", 3.0f }, { "DETUNE", 0.026f }, { "OSCGATE", 3.2f }, { "CUTOFF", 3100.0f },
                    { "FILTERENVAMOUNT", 0.22f }, { "AMPATTACK", 0.09f }, { "AMPDECAY", 0.42f }, { "AMPSUSTAIN", 0.74f },
                    { "AMPRELEASE", 0.52f }, { "FXTYPE", 2.0f }, { "FXMIX", 0.22f }, { "FXINTENSITY", 0.28f },
                    { "DELAYSEND", 0.08f }, { "REVERBMIX", 0.12f },
                });
                break;
            case 22:
                applyStringSettings ({
                    { "POLYPHONY", 24.0f }, { "UNISON", 5.0f }, { "DETUNE", 0.048f }, { "CUTOFF", 3600.0f },
                    { "FILTERENVAMOUNT", 0.18f }, { "AMPATTACK", 0.95f }, { "AMPRELEASE", 2.3f }, { "FXMIX", 0.18f },
                    { "DELAYSEND", 0.15f }, { "DELAYTIME", 0.58f }, { "DELAYFEEDBACK", 0.36f }, { "REVERBMIX", 0.22f },
                });
                break;
            case 23:
                applyStringSettings ({
                    { "POLYPHONY", 8.0f }, { "UNISON", 2.0f }, { "DETUNE", 0.018f }, { "OSCGATE", 2.4f }, { "CUTOFF", 1700.0f },
                    { "RESONANCE", 0.19f }, { "FILTERENVAMOUNT", 0.3f }, { "AMPATTACK", 0.025f }, { "AMPDECAY", 0.22f },
                    { "AMPSUSTAIN", 0.72f }, { "AMPRELEASE", 0.26f }, { "FILTATTACK", 0.001f }, { "FILTDECAY", 0.24f },
                    { "FILTSUSTAIN", 0.24f }, { "FILTRELEASE", 0.12f }, { "FXTYPE", 1.0f }, { "FXMIX", 0.05f },
                    { "FXINTENSITY", 0.12f }, { "REVERBMIX", 0.06f },
                });
                break;
            case 24:
                applyStringSettings ({
                    { "POLYPHONY", 32.0f }, { "UNISON", 6.0f }, { "DETUNE", 0.058f }, { "CUTOFF", 2900.0f },
                    { "FILTERENVAMOUNT", 0.16f }, { "AMPATTACK", 0.62f }, { "AMPRELEASE", 2.6f }, { "FXTYPE", 4.0f },
                    { "FXMIX", 0.16f }, { "FXINTENSITY", 0.32f }, { "DELAYSEND", 0.16f }, { "DELAYTIME", 0.48f },
                    { "DELAYFEEDBACK", 0.34f }, { "REVERBMIX", 0.32f },
                });
                break;
            case 25:
                applyStringSettings ({
                    { "UNISON", 4.0f }, { "DETUNE", 0.03f }, { "CUTOFF", 4800.0f }, { "RESONANCE", 0.14f },
                    { "FILTERENVAMOUNT", 0.2f }, { "AMPATTACK", 0.12f }, { "AMPDECAY", 0.45f }, { "AMPSUSTAIN", 0.82f },
                    { "AMPRELEASE", 0.9f }, { "FXTYPE", 2.0f }, { "FXMIX", 0.14f }, { "FXINTENSITY", 0.35f },
                    { "DELAYSEND", 0.1f }, { "REVERBMIX", 0.14f },
                });
                break;
            case 26:
                applyStringSettings ({
                    { "POLYPHONY", 1.0f }, { "OSCTYPE", 2.0f }, { "UNISON", 1.0f }, { "DETUNE", 0.0f }, { "OSCGATE", 4.0f },
                    { "CUTOFF", 950.0f }, { "RESONANCE", 0.24f }, { "FILTERENVAMOUNT", 0.18f }, { "AMPATTACK", 0.07f },
                    { "AMPDECAY", 0.32f }, { "AMPSUSTAIN", 0.82f }, { "AMPRELEASE", 0.6f }, { "FILTATTACK", 0.03f },
                    { "FILTDECAY", 0.5f }, { "FILTSUSTAIN", 0.4f }, { "FILTRELEASE", 0.6f }, { "FXTYPE", 0.0f },
                    { "FXMIX", 0.0f }, { "DELAYSEND", 0.0f }, { "REVERBMIX", 0.1f },
                });
                break;
            case 27:
                applyStringSettings ({
                    { "POLYPHONY", 20.0f }, { "UNISON", 4.0f }, { "DETUNE", 0.024f }, { "CUTOFF", 2300.0f },
                    { "FILTERENVAMOUNT", 0.26f }, { "AMPATTACK", 0.4f }, { "AMPDECAY", 0.9f }, { "AMPSUSTAIN", 0.76f },
                    { "AMPRELEASE", 1.7f }, { "FILTATTACK", 0.18f }, { "FILTDECAY", 1.4f }, { "FILTSUSTAIN", 0.22f },
                    { "FILTRELEASE", 1.6f }, { "FXMIX", 0.12f }, { "DELAYSEND", 0.14f }, { "REVERBMIX", 0.28f },
                });
                break;
            case 28:
                applyStringSettings ({
                    { "POLYPHONY", 48.0f }, { "UNISON", 6.0f }, { "DETUNE", 0.07f }, { "CUTOFF", 2600.0f },
                    { "FILTERENVAMOUNT", 0.12f }, { "AMPATTACK", 1.6f }, { "AMPRELEASE", 3.6f }, { "FXMIX", 0.24f },
                    { "FXINTENSITY", 0.2f }, { "DELAYSEND", 0.22f }, { "DELAYTIME", 0.72f }, { "DELAYFEEDBACK", 0.44f },
                    { "REVERBMIX", 0.38f },
                });
                break;
            case 29:
                applyStringSettings ({
                    { "POLYPHONY", 24.0f }, { "UNISON", 5.0f }, { "DETUNE", 0.055f }, { "CUTOFF", 3400.0f },
                    { "FILTERENVAMOUNT", 0.2f }, { "AMPATTACK", 0.52f }, { "AMPRELEASE", 2.0f }, { "FXTYPE", 2.0f },
                    { "FXMIX", 0.12f }, { "FXINTENSITY", 0.46f }, { "DELAYSEND", 0.12f }, { "DELAYTIME", 0.42f },
                    { "DELAYFEEDBACK", 0.3f }, { "REVERBMIX", 0.2f },
                });
                break;
            default:
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::leadSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 2100.0f);
                setParameterActual ("RESONANCE", 0.24f);
                setParameterActual ("FILTERENVAMOUNT", 0.22f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("CUTOFF", 1500.0f);
                setParameterActual ("RESONANCE", 0.18f);
                setParameterActual ("FILTERENVAMOUNT", 0.16f);
                setParameterActual ("AMPDECAY", 0.18f);
                setParameterActual ("AMPSUSTAIN", 0.78f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 1700.0f);
                setParameterActual ("RESONANCE", 0.2f);
                setParameterActual ("FILTERENVAMOUNT", 0.34f);
                setParameterActual ("AMPATTACK", 0.01f);
                setParameterActual ("AMPDECAY", 0.22f);
                setParameterActual ("AMPSUSTAIN", 0.56f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 2200.0f);
                setParameterActual ("RESONANCE", 0.22f);
                setParameterActual ("FILTERENVAMOUNT", 0.28f);
                setParameterActual ("AMPATTACK", 0.002f);
                setParameterActual ("AMPDECAY", 0.12f);
                setParameterActual ("AMPSUSTAIN", 0.7f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::padSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.03f);
                setParameterActual ("CUTOFF", 3200.0f);
                setParameterActual ("AMPATTACK", 0.62f);
                setParameterActual ("AMPRELEASE", 1.9f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.02f);
                setParameterActual ("CUTOFF", 2500.0f);
                setParameterActual ("AMPATTACK", 0.72f);
                setParameterActual ("AMPRELEASE", 2.1f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.04f);
                setParameterActual ("CUTOFF", 5200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.14f);
                setParameterActual ("AMPATTACK", 1.1f);
                setParameterActual ("AMPRELEASE", 2.6f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.04f);
                setParameterActual ("CUTOFF", 4200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                setParameterActual ("AMPATTACK", 0.5f);
                setParameterActual ("AMPRELEASE", 1.5f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::pluckSynth)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("FMAMOUNT", 48.0f);
                setParameterActual ("CUTOFF", 2800.0f);
                setParameterActual ("AMPDECAY", 0.2f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 1250.0f);
                setParameterActual ("AMPDECAY", 0.11f);
                setParameterActual ("FILTERENVAMOUNT", 0.36f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 0.0f);
                setParameterActual ("FMAMOUNT", 22.0f);
                setParameterActual ("CUTOFF", 4200.0f);
                setParameterActual ("AMPDECAY", 0.26f);
                break;
            default:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("FMAMOUNT", 0.0f);
                setParameterActual ("CUTOFF", 1800.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.44f);
                setParameterActual ("AMPDECAY", 0.16f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::sampler)
    {
        setParameterActual ("SAMPLEBANK", static_cast<float> (juce::jlimit (0, juce::jmax (0, sampleBankChoices().size() - 1), presetIndex)));
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("CUTOFF", 5400.0f);
                setParameterActual ("AMPRELEASE", 0.9f);
                break;
            case 2:
                setParameterActual ("CUTOFF", 3200.0f);
                setParameterActual ("AMPRELEASE", 0.22f);
                break;
            case 3:
                setParameterActual ("CUTOFF", 4800.0f);
                setParameterActual ("AMPRELEASE", 0.18f);
                break;
            case 4:
                setParameterActual ("CUTOFF", 1400.0f);
                setParameterActual ("AMPRELEASE", 0.2f);
                break;
            case 5:
                setParameterActual ("CUTOFF", 6200.0f);
                setParameterActual ("AMPRELEASE", 1.2f);
                break;
            default:
                setParameterActual ("CUTOFF", 4200.0f);
                setParameterActual ("AMPRELEASE", 0.36f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::acid303)
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("CUTOFF", 620.0f);
                setParameterActual ("RESONANCE", 0.74f);
                setParameterActual ("FILTERENVAMOUNT", 0.88f);
                setParameterActual ("FMAMOUNT", 360.0f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 420.0f);
                setParameterActual ("RESONANCE", 0.58f);
                setParameterActual ("FILTERENVAMOUNT", 0.62f);
                setParameterActual ("FMAMOUNT", 180.0f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 820.0f);
                setParameterActual ("RESONANCE", 0.92f);
                setParameterActual ("FILTERENVAMOUNT", 1.0f);
                setParameterActual ("FMAMOUNT", 520.0f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("CUTOFF", 540.0f);
                setParameterActual ("RESONANCE", 0.82f);
                setParameterActual ("FILTERENVAMOUNT", 0.94f);
                setParameterActual ("FMAMOUNT", 460.0f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        auto applyAdvancedSettings = [this] (std::initializer_list<std::pair<const char*, float>> settings)
        {
            for (const auto& setting : settings)
                setParameterActual (setting.first, setting.second);
        };

        auto resetAdvancedMatrix = [this]
        {
            for (int slotIndex = 1; slotIndex <= 6; ++slotIndex)
            {
                setParameterActual (("MATRIX" + juce::String (slotIndex) + "SOURCE").toRawUTF8(), 0.0f);
                for (int targetIndex = 1; targetIndex <= 3; ++targetIndex)
                {
                    setParameterActual (("MATRIX" + juce::String (slotIndex) + "AMOUNT" + juce::String (targetIndex)).toRawUTF8(), 0.0f);
                    setParameterActual (("MATRIX" + juce::String (slotIndex) + "DEST" + juce::String (targetIndex)).toRawUTF8(), 0.0f);
                }
            }
        };

        auto syncLegacyLfoRouting = [this]
        {
            setParameterActual ("LFO1AMOUNT", juce::jlimit (-1.0f, 1.0f, paramValue (apvts, "LFO1PITCH") / 24.0f));
            setParameterActual ("LFO1DEST", 0.0f);
            setParameterActual ("LFO1ASSIGNDEST", 0.0f);

            setParameterActual ("LFO2AMOUNT", juce::jlimit (-1.0f, 1.0f, paramValue (apvts, "LFO2FILTER")));
            setParameterActual ("LFO2DEST", 5.0f);
            setParameterActual ("LFO2ASSIGNDEST", 0.0f);

            setParameterActual ("LFO3AMOUNT", juce::jlimit (0.0f, 1.0f, paramValue (apvts, "RHYTHMGATE_DEPTH")));
            setParameterActual ("LFO3DEST", 10.0f);
            setParameterActual ("LFO3ASSIGNDEST", static_cast<float> (static_cast<int> (MatrixDestination::ampLevel)));
        };

        auto remapLegacyFxType = [this]
        {
            const auto legacyType = paramIndex (apvts, "FXTYPE");
            int mappedType = legacyType;
            if (legacyType == 2)
                mappedType = 4;
            else if (legacyType == 4)
                mappedType = 5;
            setParameterActual ("FXTYPE", static_cast<float> (mappedType));
        };

        resetAdvancedMatrix();
        applyAdvancedSettings ({
            { "MASTERLEVEL", 1.0f },
            { "MONOENABLE", 0.0f },
            { "ARPENABLE", 0.0f },
            { "ARPMODE", 0.0f },
            { "ARPRATE", 4.0f },
            { "OSCTYPE", 1.0f },
            { "OSC2TYPE", 2.0f },
            { "OSC3TYPE", 2.0f },
            { "UNISON", 2.0f },
            { "DETUNE", 0.045f },
            { "OSC1SEMITONE", 0.0f },
            { "OSC1DETUNE", 0.0f },
            { "OSC1PW", 0.5f },
            { "OSC2SEMITONE", 0.0f },
            { "OSC2DETUNE", 0.08f },
            { "OSC2PW", 0.5f },
            { "OSC2MIX", 0.42f },
            { "OSC3ENABLE", 1.0f },
            { "OSC3SEMITONE", -12.0f },
            { "OSC3DETUNE", 0.0f },
            { "OSC3PW", 0.5f },
            { "SUBOSCLEVEL", 0.18f },
            { "NOISELEVEL", 0.04f },
            { "RINGMOD", 0.08f },
            { "RINGMODENABLE", 1.0f },
            { "FMENABLE", 1.0f },
            { "SYNCENABLE", 1.0f },
            { "FMAMOUNT", 120.0f },
            { "SYNC", 0.24f },
            { "FILTER1ENABLE", 1.0f },
            { "FILTER2ENABLE", 1.0f },
            { "FILTERTYPE", 0.0f },
            { "FILTER2TYPE", 2.0f },
            { "FILTERSLOPE", 0.0f },
            { "FILTER2SLOPE", 0.0f },
            { "CUTOFF", 2200.0f },
            { "CUTOFF2", 5400.0f },
            { "RESONANCE", 0.36f },
            { "FILTERENVAMOUNT", 0.42f },
            { "FILTERBALANCE", 0.28f },
            { "AMPATTACK", 0.004f },
            { "AMPDECAY", 0.28f },
            { "AMPSUSTAIN", 0.74f },
            { "AMPRELEASE", 0.36f },
            { "FILTATTACK", 0.001f },
            { "FILTDECAY", 0.24f },
            { "FILTSUSTAIN", 0.42f },
            { "FILTRELEASE", 0.24f },
            { "LFO1ENABLE", 1.0f },
            { "LFO1RATE", 0.18f },
            { "LFO1SHAPE", 0.0f },
            { "LFO1ENVMODE", 0.0f },
            { "LFO1AMOUNT", 0.0f },
            { "LFO1DEST", 0.0f },
            { "LFO1ASSIGNDEST", 0.0f },
            { "LFO1PITCH", 0.0f },
            { "LFO2ENABLE", 1.0f },
            { "LFO2RATE", 0.22f },
            { "LFO2SHAPE", 0.0f },
            { "LFO2ENVMODE", 0.0f },
            { "LFO2AMOUNT", 0.12f },
            { "LFO2DEST", 5.0f },
            { "LFO2ASSIGNDEST", 0.0f },
            { "LFO2FILTER", 0.12f },
            { "LFO3ENABLE", 1.0f },
            { "LFO3SHAPE", 0.0f },
            { "LFO3ENVMODE", 1.0f },
            { "LFO3AMOUNT", 0.0f },
            { "LFO3DEST", 10.0f },
            { "LFO3ASSIGNDEST", static_cast<float> (static_cast<int> (MatrixDestination::ampLevel)) },
            { "RHYTHMGATE_RATE", 8.0f },
            { "RHYTHMGATE_DEPTH", 0.0f },
            { "FXTYPE", 3.0f },
            { "FXMIX", 0.18f },
            { "FXINTENSITY", 0.32f },
            { "DELAYSEND", 0.16f },
            { "DELAYTIME", 0.36f },
            { "DELAYFEEDBACK", 0.28f },
            { "REVERBMIX", 0.12f },
            { "REVERBTIME", 0.48f },
            { "REVERBDAMP", 0.4f },
            { "LOWEQGAIN", 0.0f },
            { "LOWEQFREQ", 220.0f },
            { "LOWEQQ", 0.8f },
            { "MIDEQGAIN", 0.0f },
            { "MIDEQFREQ", 1400.0f },
            { "MIDEQQ", 1.1f },
            { "HIGHEQGAIN", 0.0f },
            { "HIGHEQFREQ", 5000.0f },
            { "HIGHEQQ", 0.8f },
        });

        const auto builtInCount = builtInAdvancedVirusPresetChoices().size();
        const auto importedIndex = presetIndex - builtInCount;
        const auto& importedPresets = importedVirusPresets();

        if (juce::isPositiveAndBelow (importedIndex, static_cast<int> (importedPresets.size())))
        {
            const auto& imported = importedPresets[static_cast<size_t> (importedIndex)];
            const auto lowerName = imported.name.toLowerCase();

            const auto pitchSeed = importedPayloadNorm (imported, 18);
            const auto colourSeed = importedPayloadNorm (imported, 36);
            const auto motionSeed = importedPayloadNorm (imported, 64);
            const auto filterSeed = importedPayloadNorm (imported, 82);
            const auto ampSeed = importedPayloadNorm (imported, 101);
            const auto fxSeed = importedPayloadNorm (imported, 154);
            const auto envSeed = importedPayloadNorm (imported, 188);

            const bool isBass = importedNameContains (lowerName, { "bass", "sub", "303", "acid" });
            const bool isLead = importedNameContains (lowerName, { "lead", "lazer", "solo", "hero", "gallop", "zipp" });
            const bool isPad = importedNameContains (lowerName, { "pad", "wash", "choir", "solar", "omni", "mystery", "swell" });
            const bool isArp = importedNameContains (lowerName, { "arp", "seq", "gallop", "pulse", "step" });
            const bool isFm = importedNameContains (lowerName, { "fm", "sin", "metal", "bell" });
            const bool isNoise = importedNameContains (lowerName, { "noise", "dirty", "chip", "ufo", "rev" });
            const bool isWide = importedNameContains (lowerName, { "wide", "omni", "solar", "wave", "pad", "poly" });
            const bool isBright = importedNameContains (lowerName, { "high", "glass", "chip", "bright", "air" });
            const bool isDark = importedNameContains (lowerName, { "deep", "dark", "mystery" });

            int osc1Type = 1;
            int osc2Type = 1;

            if (importedNameContains (lowerName, { "square", "chip" }))
                osc1Type = 2;
            else if (isFm)
                osc1Type = 0;
            else if (isPad)
                osc1Type = 5 + static_cast<int> (std::floor (colourSeed * 4.0f));

            if (importedNameContains (lowerName, { "vocal", "choir" }))
                osc2Type = 9;
            else if (importedNameContains (lowerName, { "form", "talk" }))
                osc2Type = 6;
            else if (importedNameContains (lowerName, { "metal" }))
                osc2Type = 8;
            else if (isFm)
                osc2Type = 0;
            else if (isPad)
                osc2Type = 7 + static_cast<int> (std::round (pitchSeed));
            else if (importedNameContains (lowerName, { "square", "chip" }))
                osc2Type = 2;

            osc1Type = juce::jlimit (0, 9, osc1Type);
            osc2Type = juce::jlimit (0, 9, osc2Type);

            const float cutoff = isBass ? importedRange (imported, 120, 120.0f, 1800.0f)
                                        : (isPad ? importedRange (imported, 120, 1400.0f, 5200.0f)
                                                 : importedRange (imported, 120, 700.0f, 7200.0f));
            const float cutoff2 = juce::jmax (cutoff + 120.0f, importedRange (imported, 127, 1800.0f, 12000.0f));
            const float resonance = importedRange (imported, 128, isBass ? 0.18f : 0.12f, isLead ? 0.72f : 0.56f);
            const float envAmount = importedRange (imported, 132, isPad ? 0.08f : 0.16f, isLead ? 0.8f : 0.62f);
            const float ampAttack = isPad ? importedRange (imported, 138, 0.08f, 1.8f)
                                          : importedRange (imported, 138, 0.001f, isBass ? 0.08f : 0.28f);
            const float ampDecay = importedRange (imported, 140, isPad ? 0.45f : 0.08f, isPad ? 1.9f : 0.95f);
            const float ampSustain = importedRange (imported, 142, isArp ? 0.05f : 0.32f, isPad ? 0.96f : 0.84f);
            const float ampRelease = importedRange (imported, 144, isArp ? 0.04f : 0.12f, isPad ? 2.6f : 0.9f);
            const float filtAttack = importedRange (imported, 146, 0.001f, isPad ? 0.5f : 0.1f);
            const float filtDecay = importedRange (imported, 148, 0.06f, isPad ? 1.6f : 0.7f);
            const float filtSustain = importedRange (imported, 150, 0.04f, isPad ? 0.72f : 0.44f);
            const float filtRelease = importedRange (imported, 152, 0.04f, isPad ? 1.8f : 0.8f);

            const int fxType = isFm ? 6 : (isPad ? 3 : (isNoise ? 1 : ((fxSeed > 0.72f) ? 5 : 0)));
            const float fxMix = importedRange (imported, 168, 0.04f, isPad ? 0.34f : 0.24f);
            const float fxIntensity = importedRange (imported, 170, 0.12f, isFm ? 0.72f : 0.5f);
            const float fxRate = importedRange (imported, 171, 0.08f, 7.2f);
            const float fxColour = importedRange (imported, 172, 0.18f, 0.9f);
            const float fxSpread = importedRange (imported, 173, 0.1f, 0.95f);
            const float delaySend = importedRange (imported, 174, 0.0f, isPad ? 0.26f : 0.14f);
            const float delayTime = importedRange (imported, 175, 0.18f, 0.72f);
            const float delayFeedback = importedRange (imported, 176, 0.06f, 0.48f);
            const float reverbMix = importedRange (imported, 177, 0.02f, isPad ? 0.34f : 0.18f);
            const float reverbTime = importedRange (imported, 178, 0.24f, 0.92f);
            const float reverbDamp = importedRange (imported, 179, 0.2f, 0.82f);

            const float lfoAmount = importedRange (imported, 194, isPad ? 0.04f : 0.0f, isPad ? 0.22f : 0.14f);
            const float lfoRate = importedRange (imported, 196, 0.08f, isArp ? 8.0f : 2.0f);

            applyAdvancedSettings ({
                { "MONOENABLE", isBass || isLead ? 1.0f : 0.0f },
                { "ARPENABLE", isArp ? 1.0f : 0.0f },
                { "ARPMODE", isArp ? static_cast<float> (juce::jlimit (0, 3, 1 + static_cast<int> (std::floor (motionSeed * 3.0f)))) : 0.0f },
                { "ARPRATE", isArp ? importedRange (imported, 210, 4.0f, 12.0f) : 4.0f },
                { "OSCTYPE", static_cast<float> (osc1Type) },
                { "OSC2TYPE", static_cast<float> (osc2Type) },
                { "UNISON", isWide ? 2.0f : 1.0f },
                { "DETUNE", importedRange (imported, 114, isBass ? 0.006f : 0.012f, isWide ? 0.09f : 0.05f) },
                { "OSC1SEMITONE", 0.0f },
                { "OSC1DETUNE", importedRange (imported, 115, -0.05f, 0.05f) },
                { "OSC1PW", importedRange (imported, 116, 0.22f, 0.82f) },
                { "OSC2SEMITONE", isBass ? -12.0f : (isBright ? 12.0f : (importedPayloadNorm (imported, 117) > 0.66f ? 7.0f : 0.0f)) },
                { "OSC2DETUNE", importedRange (imported, 118, -0.12f, 0.12f) },
                { "OSC2PW", importedRange (imported, 119, 0.22f, 0.82f) },
                { "OSC2MIX", importedRange (imported, 121, 0.18f, 0.58f) },
                { "OSC3ENABLE", importedPayloadNorm (imported, 122) > 0.28f ? 1.0f : 0.0f },
                { "OSC3SEMITONE", isBass ? -12.0f : 0.0f },
                { "OSC3DETUNE", importedRange (imported, 123, -0.05f, 0.05f) },
                { "OSC3PW", importedRange (imported, 124, 0.24f, 0.76f) },
                { "SUBOSCLEVEL", importedRange (imported, 125, isBass ? 0.18f : 0.0f, isBass ? 0.62f : 0.24f) },
                { "NOISELEVEL", importedRange (imported, 126, 0.0f, isNoise ? 0.34f : 0.12f) },
                { "RINGMOD", importedRange (imported, 129, 0.0f, isFm ? 0.42f : 0.18f) },
                { "FMAMOUNT", importedRange (imported, 130, isFm ? 90.0f : 0.0f, isFm ? 760.0f : 220.0f) },
                { "SYNC", importedRange (imported, 131, 0.0f, isLead ? 0.66f : 0.28f) },
                { "FILTER1ENABLE", 1.0f },
                { "FILTER2ENABLE", importedPayloadNorm (imported, 133) > 0.22f ? 1.0f : 0.0f },
                { "FILTERTYPE", static_cast<float> (juce::jlimit (0, 3, static_cast<int> (std::floor (filterSeed * 4.0f)))) },
                { "FILTER2TYPE", static_cast<float> (juce::jlimit (0, 3, static_cast<int> (std::floor (importedPayloadNorm (imported, 134) * 4.0f)))) },
                { "CUTOFF", cutoff },
                { "CUTOFF2", cutoff2 },
                { "RESONANCE", resonance },
                { "FILTERENVAMOUNT", envAmount },
                { "FILTERBALANCE", importedRange (imported, 135, 0.08f, 0.52f) },
                { "PANORAMA", importedRange (imported, 136, -0.35f, 0.35f) },
                { "KEYFOLLOW", importedRange (imported, 137, isDark ? 0.06f : 0.18f, 0.82f) },
                { "AMPATTACK", ampAttack },
                { "AMPDECAY", ampDecay },
                { "AMPSUSTAIN", ampSustain },
                { "AMPRELEASE", ampRelease },
                { "FILTATTACK", filtAttack },
                { "FILTDECAY", filtDecay },
                { "FILTSUSTAIN", filtSustain },
                { "FILTRELEASE", filtRelease },
                { "LFO1ENABLE", lfoAmount > 0.01f ? 1.0f : 0.0f },
                { "LFO1RATE", lfoRate },
                { "LFO1SHAPE", static_cast<float> (juce::jlimit (0, 4, static_cast<int> (std::floor (importedPayloadNorm (imported, 198) * 5.0f)))) },
                { "LFO1ENVMODE", isArp ? 1.0f : 0.0f },
                { "LFO1AMOUNT", lfoAmount },
                { "LFO1DEST", static_cast<float> (juce::jlimit (0, 10, static_cast<int> (std::floor (importedPayloadNorm (imported, 199) * 11.0f)))) },
                { "LFO1ASSIGNDEST", static_cast<float> (static_cast<int> (MatrixDestination::off)) },
                { "LFO2ENABLE", importedPayloadNorm (imported, 200) > 0.18f ? 1.0f : 0.0f },
                { "LFO2RATE", importedRange (imported, 201, 0.08f, 6.2f) },
                { "LFO2SHAPE", static_cast<float> (juce::jlimit (0, 4, static_cast<int> (std::floor (importedPayloadNorm (imported, 202) * 5.0f)))) },
                { "LFO2ENVMODE", isPad ? 0.0f : 1.0f },
                { "LFO2AMOUNT", importedRange (imported, 203, 0.0f, 0.24f) },
                { "LFO2DEST", static_cast<float> (juce::jlimit (0, 10, static_cast<int> (std::floor (importedPayloadNorm (imported, 204) * 11.0f)))) },
                { "LFO2ASSIGNDEST", static_cast<float> (static_cast<int> (MatrixDestination::off)) },
                { "LFO3ENABLE", importedPayloadNorm (imported, 205) > 0.36f ? 1.0f : 0.0f },
                { "RHYTHMGATE_RATE", importedRange (imported, 206, 2.0f, 12.0f) },
                { "LFO3SHAPE", static_cast<float> (juce::jlimit (0, 4, static_cast<int> (std::floor (importedPayloadNorm (imported, 207) * 5.0f)))) },
                { "LFO3ENVMODE", 1.0f },
                { "LFO3AMOUNT", importedRange (imported, 208, 0.0f, 0.26f) },
                { "LFO3DEST", static_cast<float> (juce::jlimit (0, 10, static_cast<int> (std::floor (importedPayloadNorm (imported, 209) * 11.0f)))) },
                { "LFO3ASSIGNDEST", static_cast<float> (static_cast<int> (MatrixDestination::assign)) },
                { "FXTYPE", static_cast<float> (fxType) },
                { "FXMIX", fxMix },
                { "FXINTENSITY", fxIntensity },
                { "FXRATE", fxRate },
                { "FXCOLOUR", fxColour },
                { "FXSPREAD", fxSpread },
                { "DELAYSEND", delaySend },
                { "DELAYTIME", delayTime },
                { "DELAYFEEDBACK", delayFeedback },
                { "REVERBMIX", reverbMix },
                { "REVERBTIME", reverbTime },
                { "REVERBDAMP", reverbDamp },
                { "LOWEQGAIN", importedRange (imported, 180, -4.0f, 4.0f) },
                { "LOWEQFREQ", importedRange (imported, 181, 90.0f, 420.0f) },
                { "LOWEQQ", importedRange (imported, 182, 0.5f, 1.4f) },
                { "MIDEQGAIN", importedRange (imported, 183, -4.0f, 4.0f) },
                { "MIDEQFREQ", importedRange (imported, 184, 420.0f, 4200.0f) },
                { "MIDEQQ", importedRange (imported, 185, 0.6f, 1.7f) },
                { "HIGHEQGAIN", importedRange (imported, 186, -4.0f, 4.0f) },
                { "HIGHEQFREQ", importedRange (imported, 187, 1800.0f, 9000.0f) },
                { "HIGHEQQ", importedRange (imported, 189, 0.5f, 1.4f) },
                { "SATURATIONTYPE", static_cast<float> (juce::jlimit (0, 3, static_cast<int> (std::floor (colourSeed * 4.0f)))) },
                { "MASTERLEVEL", importedRange (imported, 190, 0.9f, isBass ? 1.18f : 1.06f) },
            });

            resetAdvancedMatrix();
            if (motionSeed > 0.42f)
            {
                setParameterActual ("MATRIX1SOURCE", 1.0f);
                setParameterActual ("MATRIX1DEST1", static_cast<float> (static_cast<int> (MatrixDestination::cutoff1)));
                setParameterActual ("MATRIX1AMOUNT1", importedRange (imported, 220, -0.12f, 0.22f));
                if (isWide)
                {
                    setParameterActual ("MATRIX1DEST2", static_cast<float> (static_cast<int> (MatrixDestination::panorama)));
                    setParameterActual ("MATRIX1AMOUNT2", importedRange (imported, 221, 0.02f, 0.18f));
                }
            }

            if (envSeed > 0.55f)
            {
                setParameterActual ("MATRIX2SOURCE", 5.0f);
                setParameterActual ("MATRIX2DEST1", static_cast<float> (static_cast<int> (MatrixDestination::filterBalance)));
                setParameterActual ("MATRIX2AMOUNT1", importedRange (imported, 222, -0.14f, 0.22f));
            }

            remapLegacyFxType();
            syncLegacyLfoRouting();
            return;
        }

        switch (presetIndex)
        {
            case 1:
                applyAdvancedSettings ({
                    { "OSCTYPE", 2.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 2.0f }, { "DETUNE", 0.03f },
                    { "OSC2DETUNE", 0.04f }, { "OSC2MIX", 0.36f }, { "SUBOSCLEVEL", 0.14f }, { "NOISELEVEL", 0.02f },
                    { "RINGMOD", 0.04f }, { "FMAMOUNT", 80.0f }, { "SYNC", 0.18f }, { "CUTOFF", 1800.0f },
                    { "CUTOFF2", 4200.0f }, { "RESONANCE", 0.3f }, { "FILTERENVAMOUNT", 0.34f },
                    { "FILTERBALANCE", 0.22f }, { "AMPDECAY", 0.24f }, { "AMPSUSTAIN", 0.78f },
                    { "AMPRELEASE", 0.32f }, { "FILTDECAY", 0.18f }, { "FILTSUSTAIN", 0.34f },
                    { "FILTRELEASE", 0.18f }, { "FXMIX", 0.16f }, { "FXINTENSITY", 0.28f },
                    { "DELAYSEND", 0.12f }, { "DELAYTIME", 0.32f }, { "DELAYFEEDBACK", 0.22f },
                    { "REVERBMIX", 0.08f },
                });
                break;
            case 2:
                applyAdvancedSettings ({
                    { "DETUNE", 0.06f }, { "OSC2SEMITONE", 12.0f }, { "OSC2MIX", 0.52f },
                    { "SUBOSCLEVEL", 0.08f }, { "NOISELEVEL", 0.06f }, { "RINGMOD", 0.18f },
                    { "FMAMOUNT", 160.0f }, { "SYNC", 0.36f }, { "FILTERTYPE", 1.0f },
                    { "CUTOFF", 3200.0f }, { "CUTOFF2", 7600.0f }, { "RESONANCE", 0.42f },
                    { "FILTERENVAMOUNT", 0.52f }, { "FILTERBALANCE", 0.34f }, { "FXTYPE", 2.0f },
                    { "FXMIX", 0.28f }, { "FXINTENSITY", 0.42f }, { "DELAYSEND", 0.18f },
                    { "DELAYTIME", 0.44f }, { "DELAYFEEDBACK", 0.3f },
                });
                break;
            case 3:
                applyAdvancedSettings ({
                    { "OSCTYPE", 0.0f }, { "OSC2TYPE", 4.0f }, { "UNISON", 1.0f }, { "DETUNE", 0.02f },
                    { "OSC2SEMITONE", 7.0f }, { "OSC2DETUNE", -0.06f }, { "OSC2MIX", 0.28f },
                    { "SUBOSCLEVEL", 0.22f }, { "NOISELEVEL", 0.08f }, { "RINGMOD", 0.0f },
                    { "FMAMOUNT", 44.0f }, { "SYNC", 0.1f }, { "FILTER2TYPE", 1.0f }, { "CUTOFF", 980.0f },
                    { "CUTOFF2", 2400.0f }, { "RESONANCE", 0.24f }, { "FILTERENVAMOUNT", 0.24f },
                    { "FILTERBALANCE", 0.12f }, { "FXTYPE", 1.0f }, { "FXMIX", 0.14f },
                    { "FXINTENSITY", 0.36f }, { "DELAYSEND", 0.06f }, { "DELAYTIME", 0.26f },
                    { "DELAYFEEDBACK", 0.18f }, { "REVERBMIX", 0.04f },
                });
                break;
            case 4:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "OSCTYPE", 2.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 1.0f },
                    { "DETUNE", 0.01f }, { "OSC2DETUNE", 0.02f }, { "OSC2MIX", 0.22f }, { "SUBOSCLEVEL", 0.38f },
                    { "NOISELEVEL", 0.0f }, { "RINGMOD", 0.0f }, { "FMAMOUNT", 28.0f }, { "SYNC", 0.06f },
                    { "CUTOFF", 340.0f }, { "CUTOFF2", 1800.0f }, { "RESONANCE", 0.62f }, { "FILTERENVAMOUNT", 0.78f },
                    { "FILTERBALANCE", 0.16f }, { "AMPATTACK", 0.001f }, { "AMPDECAY", 0.18f },
                    { "AMPSUSTAIN", 0.46f }, { "AMPRELEASE", 0.1f }, { "FILTDECAY", 0.2f },
                    { "FILTSUSTAIN", 0.08f }, { "FILTRELEASE", 0.16f }, { "FXTYPE", 1.0f }, { "FXMIX", 0.1f },
                    { "FXINTENSITY", 0.34f }, { "DELAYSEND", 0.02f }, { "REVERBMIX", 0.0f }, { "MASTERLEVEL", 0.95f },
                });
                break;
            case 5:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "ARPENABLE", 1.0f }, { "ARPMODE", 2.0f }, { "ARPRATE", 8.0f },
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 2.0f }, { "DETUNE", 0.03f }, { "OSC2SEMITONE", 12.0f },
                    { "OSC2MIX", 0.36f }, { "SUBOSCLEVEL", 0.08f }, { "NOISELEVEL", 0.05f }, { "RINGMOD", 0.14f },
                    { "FMAMOUNT", 90.0f }, { "SYNC", 0.22f }, { "FILTERTYPE", 1.0f }, { "CUTOFF", 1500.0f },
                    { "CUTOFF2", 5200.0f }, { "RESONANCE", 0.72f }, { "FILTERENVAMOUNT", 0.68f },
                    { "FILTERBALANCE", 0.38f }, { "AMPDECAY", 0.22f }, { "AMPSUSTAIN", 0.22f },
                    { "AMPRELEASE", 0.08f }, { "FILTDECAY", 0.14f }, { "FILTSUSTAIN", 0.04f },
                    { "LFO2RATE", 5.2f }, { "LFO2FILTER", 0.16f }, { "FXTYPE", 2.0f }, { "FXMIX", 0.24f },
                    { "FXINTENSITY", 0.36f }, { "DELAYSEND", 0.1f }, { "DELAYTIME", 0.29f },
                    { "DELAYFEEDBACK", 0.18f }, { "REVERBMIX", 0.04f },
                });
                break;
            case 6:
                applyAdvancedSettings ({
                    { "ARPENABLE", 1.0f }, { "ARPMODE", 0.0f }, { "ARPRATE", 6.0f }, { "OSCTYPE", 1.0f },
                    { "OSC2TYPE", 1.0f }, { "DETUNE", 0.02f }, { "OSC2SEMITONE", 7.0f }, { "OSC2MIX", 0.32f },
                    { "SUBOSCLEVEL", 0.12f }, { "NOISELEVEL", 0.01f }, { "FMAMOUNT", 30.0f }, { "SYNC", 0.1f },
                    { "CUTOFF", 1200.0f }, { "CUTOFF2", 3800.0f }, { "RESONANCE", 0.38f }, { "FILTERENVAMOUNT", 0.42f },
                    { "AMPDECAY", 0.3f }, { "AMPSUSTAIN", 0.5f }, { "AMPRELEASE", 0.15f }, { "FXMIX", 0.14f },
                    { "FXINTENSITY", 0.3f }, { "DELAYSEND", 0.18f }, { "DELAYTIME", 0.36f },
                    { "DELAYFEEDBACK", 0.24f }, { "REVERBMIX", 0.08f },
                });
                break;
            case 7:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "ARPENABLE", 1.0f }, { "ARPMODE", 3.0f }, { "ARPRATE", 10.0f },
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 2.0f }, { "DETUNE", 0.04f }, { "OSC2SEMITONE", 12.0f },
                    { "OSC2MIX", 0.46f }, { "NOISELEVEL", 0.12f }, { "RINGMOD", 0.22f }, { "FMAMOUNT", 180.0f },
                    { "SYNC", 0.4f }, { "FILTERTYPE", 1.0f }, { "CUTOFF", 2800.0f }, { "CUTOFF2", 7800.0f },
                    { "RESONANCE", 0.66f }, { "FILTERENVAMOUNT", 0.54f }, { "FILTERBALANCE", 0.48f },
                    { "AMPDECAY", 0.18f }, { "AMPSUSTAIN", 0.16f }, { "AMPRELEASE", 0.06f }, { "FILTDECAY", 0.12f },
                    { "FILTSUSTAIN", 0.06f }, { "LFO2RATE", 7.0f }, { "LFO2FILTER", 0.22f }, { "FXTYPE", 4.0f },
                    { "FXMIX", 0.28f }, { "FXINTENSITY", 0.52f }, { "DELAYSEND", 0.18f }, { "DELAYTIME", 0.41f },
                    { "DELAYFEEDBACK", 0.28f }, { "REVERBMIX", 0.06f },
                });
                break;
            case 8:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "OSCTYPE", 0.0f }, { "OSC2TYPE", 0.0f }, { "UNISON", 1.0f },
                    { "DETUNE", 0.0f }, { "OSC2SEMITONE", 12.0f }, { "OSC2MIX", 0.18f }, { "SUBOSCLEVEL", 0.0f },
                    { "NOISELEVEL", 0.05f }, { "RINGMOD", 0.0f }, { "FMAMOUNT", 420.0f }, { "SYNC", 0.0f },
                    { "CUTOFF", 6400.0f }, { "CUTOFF2", 12000.0f }, { "RESONANCE", 0.24f }, { "FILTERENVAMOUNT", 0.82f },
                    { "FILTERBALANCE", 0.1f }, { "AMPDECAY", 0.06f }, { "AMPSUSTAIN", 0.0f }, { "AMPRELEASE", 0.04f },
                    { "FILTDECAY", 0.08f }, { "FILTSUSTAIN", 0.0f }, { "FILTRELEASE", 0.05f }, { "FXTYPE", 0.0f },
                    { "FXMIX", 0.0f }, { "FXINTENSITY", 0.0f }, { "DELAYSEND", 0.0f }, { "REVERBMIX", 0.05f },
                });
                break;
            case 9:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "OSCTYPE", 0.0f }, { "OSC2TYPE", 0.0f }, { "UNISON", 1.0f },
                    { "OSC2SEMITONE", 19.0f }, { "OSC2MIX", 0.12f }, { "SUBOSCLEVEL", 0.0f }, { "NOISELEVEL", 0.0f },
                    { "RINGMOD", 0.0f }, { "FMAMOUNT", 760.0f }, { "SYNC", 0.0f }, { "FILTERTYPE", 1.0f },
                    { "FILTER2ENABLE", 0.0f }, { "CUTOFF", 3200.0f }, { "CUTOFF2", 4200.0f }, { "RESONANCE", 0.68f },
                    { "FILTERENVAMOUNT", 0.48f }, { "FILTERBALANCE", 0.0f }, { "AMPDECAY", 0.035f },
                    { "AMPSUSTAIN", 0.0f }, { "AMPRELEASE", 0.05f }, { "FILTDECAY", 0.05f }, { "FILTSUSTAIN", 0.0f },
                    { "FILTRELEASE", 0.05f }, { "FXTYPE", 0.0f }, { "FXMIX", 0.0f }, { "FXINTENSITY", 0.0f },
                    { "DELAYSEND", 0.0f }, { "REVERBMIX", 0.0f }, { "MASTERLEVEL", 1.1f },
                });
                break;
            case 10:
                applyAdvancedSettings ({
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 6.0f }, { "DETUNE", 0.08f },
                    { "OSC2DETUNE", 0.12f }, { "OSC2MIX", 0.48f }, { "SUBOSCLEVEL", 0.12f }, { "NOISELEVEL", 0.02f },
                    { "RINGMOD", 0.0f }, { "FILTER2ENABLE", 0.0f }, { "CUTOFF", 1800.0f }, { "RESONANCE", 0.22f },
                    { "FILTERENVAMOUNT", 0.14f }, { "FILTERBALANCE", 0.12f }, { "AMPATTACK", 0.8f },
                    { "AMPDECAY", 1.2f }, { "AMPSUSTAIN", 0.88f }, { "AMPRELEASE", 2.8f }, { "FILTATTACK", 0.3f },
                    { "FILTDECAY", 1.8f }, { "FILTSUSTAIN", 0.54f }, { "FILTRELEASE", 1.8f }, { "LFO2RATE", 0.18f },
                    { "LFO2SHAPE", 1.0f }, { "LFO2FILTER", 0.18f }, { "FXTYPE", 3.0f }, { "FXMIX", 0.28f },
                    { "FXINTENSITY", 0.4f }, { "DELAYSEND", 0.24f }, { "DELAYTIME", 0.58f },
                    { "DELAYFEEDBACK", 0.42f }, { "REVERBMIX", 0.32f },
                });
                break;
            case 11:
                applyAdvancedSettings ({
                    { "OSCTYPE", 3.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 1.0f }, { "DETUNE", 0.02f },
                    { "OSC2SEMITONE", -12.0f }, { "OSC2MIX", 0.22f }, { "SUBOSCLEVEL", 0.0f }, { "NOISELEVEL", 0.28f },
                    { "RINGMOD", 0.06f }, { "FMAMOUNT", 40.0f }, { "SYNC", 0.0f }, { "FILTERTYPE", 3.0f },
                    { "CUTOFF", 3200.0f }, { "CUTOFF2", 6800.0f }, { "RESONANCE", 0.74f }, { "FILTERENVAMOUNT", 0.22f },
                    { "FILTERBALANCE", 0.52f }, { "AMPATTACK", 0.42f }, { "AMPDECAY", 0.7f }, { "AMPSUSTAIN", 0.38f },
                    { "AMPRELEASE", 1.1f }, { "FILTATTACK", 0.58f }, { "FILTDECAY", 1.6f }, { "FILTSUSTAIN", 0.22f },
                    { "FILTRELEASE", 1.0f }, { "LFO2RATE", 0.34f }, { "LFO2SHAPE", 2.0f }, { "LFO2FILTER", 0.28f },
                    { "FXTYPE", 2.0f }, { "FXMIX", 0.32f }, { "FXINTENSITY", 0.62f }, { "DELAYSEND", 0.1f },
                    { "DELAYTIME", 0.47f }, { "DELAYFEEDBACK", 0.36f }, { "REVERBMIX", 0.22f },
                });
                break;
            case 12:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "ARPENABLE", 1.0f }, { "ARPMODE", 2.0f }, { "ARPRATE", 12.0f },
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 2.0f }, { "UNISON", 1.0f }, { "DETUNE", 0.015f },
                    { "OSC2SEMITONE", 12.0f }, { "OSC2MIX", 0.3f }, { "SUBOSCLEVEL", 0.0f }, { "NOISELEVEL", 0.02f },
                    { "RINGMOD", 0.04f }, { "FMAMOUNT", 220.0f }, { "SYNC", 0.55f }, { "FILTERTYPE", 1.0f },
                    { "CUTOFF", 2400.0f }, { "CUTOFF2", 7600.0f }, { "RESONANCE", 0.56f }, { "FILTERENVAMOUNT", 0.64f },
                    { "FILTERBALANCE", 0.46f }, { "AMPDECAY", 0.14f }, { "AMPSUSTAIN", 0.12f }, { "AMPRELEASE", 0.08f },
                    { "FILTDECAY", 0.12f }, { "FILTSUSTAIN", 0.06f }, { "FXTYPE", 4.0f }, { "FXMIX", 0.18f },
                    { "FXINTENSITY", 0.44f }, { "DELAYSEND", 0.08f }, { "REVERBMIX", 0.04f },
                });
                break;
            case 13:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "OSCTYPE", 1.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 1.0f },
                    { "DETUNE", 0.0f }, { "OSC2DETUNE", 0.02f }, { "OSC2MIX", 0.24f }, { "SUBOSCLEVEL", 0.52f },
                    { "NOISELEVEL", 0.0f }, { "RINGMOD", 0.0f }, { "FMAMOUNT", 16.0f }, { "SYNC", 0.0f },
                    { "FILTER2ENABLE", 0.0f }, { "CUTOFF", 180.0f }, { "RESONANCE", 0.22f }, { "FILTERENVAMOUNT", 0.42f },
                    { "AMPATTACK", 0.001f }, { "AMPDECAY", 0.22f }, { "AMPSUSTAIN", 0.54f }, { "AMPRELEASE", 0.12f },
                    { "FILTDECAY", 0.2f }, { "FILTSUSTAIN", 0.08f }, { "FXTYPE", 1.0f }, { "FXMIX", 0.06f },
                    { "FXINTENSITY", 0.28f }, { "DELAYSEND", 0.0f }, { "REVERBMIX", 0.0f }, { "MASTERLEVEL", 1.1f },
                });
                break;
            case 14:
                applyAdvancedSettings ({
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 7.0f }, { "DETUNE", 0.09f },
                    { "OSC2MIX", 0.5f }, { "SUBOSCLEVEL", 0.12f }, { "NOISELEVEL", 0.03f }, { "RINGMOD", 0.0f },
                    { "FILTER2ENABLE", 0.0f }, { "CUTOFF", 4200.0f }, { "RESONANCE", 0.18f }, { "FILTERENVAMOUNT", 0.18f },
                    { "AMPATTACK", 0.9f }, { "AMPDECAY", 1.2f }, { "AMPSUSTAIN", 0.92f }, { "AMPRELEASE", 3.0f },
                    { "FILTATTACK", 0.24f }, { "FILTDECAY", 1.1f }, { "FILTSUSTAIN", 0.64f }, { "FILTRELEASE", 1.8f },
                    { "LFO2RATE", 0.12f }, { "LFO2SHAPE", 1.0f }, { "LFO2FILTER", 0.14f }, { "FXTYPE", 3.0f },
                    { "FXMIX", 0.26f }, { "FXINTENSITY", 0.34f }, { "DELAYSEND", 0.18f }, { "DELAYTIME", 0.52f },
                    { "DELAYFEEDBACK", 0.38f }, { "REVERBMIX", 0.26f },
                });
                break;
            case 15:
                applyAdvancedSettings ({
                    { "MONOENABLE", 1.0f }, { "OSCTYPE", 2.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 2.0f },
                    { "DETUNE", 0.028f }, { "OSC2SEMITONE", 7.0f }, { "OSC2MIX", 0.34f }, { "SUBOSCLEVEL", 0.08f },
                    { "NOISELEVEL", 0.02f }, { "RINGMOD", 0.0f }, { "FMAMOUNT", 110.0f }, { "SYNC", 0.18f },
                    { "FILTERTYPE", 1.0f }, { "CUTOFF", 1100.0f }, { "CUTOFF2", 4200.0f }, { "RESONANCE", 0.52f },
                    { "FILTERENVAMOUNT", 0.58f }, { "FILTERBALANCE", 0.34f }, { "AMPATTACK", 0.01f },
                    { "AMPDECAY", 0.18f }, { "AMPSUSTAIN", 0.36f }, { "AMPRELEASE", 0.14f }, { "FILTDECAY", 0.16f },
                    { "FILTSUSTAIN", 0.08f }, { "FXTYPE", 2.0f }, { "FXMIX", 0.16f }, { "FXINTENSITY", 0.44f },
                    { "DELAYSEND", 0.06f }, { "REVERBMIX", 0.04f },
                });
                break;
            case 16:
                applyAdvancedSettings ({
                    { "OSCTYPE", 0.0f }, { "OSC2TYPE", 0.0f }, { "UNISON", 2.0f }, { "DETUNE", 0.016f },
                    { "OSC2SEMITONE", 7.0f }, { "OSC2MIX", 0.36f }, { "SUBOSCLEVEL", 0.0f }, { "NOISELEVEL", 0.0f },
                    { "RINGMOD", 0.04f }, { "FMAMOUNT", 260.0f }, { "SYNC", 0.0f }, { "FILTER2ENABLE", 0.0f },
                    { "CUTOFF", 3600.0f }, { "RESONANCE", 0.16f }, { "FILTERENVAMOUNT", 0.28f }, { "AMPATTACK", 0.02f },
                    { "AMPDECAY", 0.62f }, { "AMPSUSTAIN", 0.68f }, { "AMPRELEASE", 0.74f }, { "FILTATTACK", 0.01f },
                    { "FILTDECAY", 0.38f }, { "FILTSUSTAIN", 0.42f }, { "FILTRELEASE", 0.44f }, { "FXTYPE", 3.0f },
                    { "FXMIX", 0.12f }, { "FXINTENSITY", 0.22f }, { "DELAYSEND", 0.08f }, { "REVERBMIX", 0.18f },
                });
                break;
            case 17:
                applyAdvancedSettings ({
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 0.0f }, { "UNISON", 5.0f }, { "DETUNE", 0.055f },
                    { "OSC2SEMITONE", 12.0f }, { "OSC2MIX", 0.42f }, { "SUBOSCLEVEL", 0.12f }, { "NOISELEVEL", 0.02f },
                    { "RINGMOD", 0.0f }, { "CUTOFF", 2400.0f }, { "CUTOFF2", 5200.0f }, { "RESONANCE", 0.24f },
                    { "FILTERENVAMOUNT", 0.16f }, { "FILTERBALANCE", 0.2f }, { "AMPATTACK", 0.36f },
                    { "AMPDECAY", 0.92f }, { "AMPSUSTAIN", 0.84f }, { "AMPRELEASE", 2.2f }, { "FILTATTACK", 0.12f },
                    { "FILTDECAY", 0.76f }, { "FILTSUSTAIN", 0.52f }, { "FILTRELEASE", 1.1f }, { "LFO2RATE", 0.09f },
                    { "LFO2FILTER", 0.12f }, { "FXTYPE", 3.0f }, { "FXMIX", 0.22f }, { "FXINTENSITY", 0.3f },
                    { "DELAYSEND", 0.14f }, { "DELAYTIME", 0.48f }, { "DELAYFEEDBACK", 0.32f },
                    { "REVERBMIX", 0.26f },
                });
                break;
            case 18:
                applyAdvancedSettings ({
                    { "OSCTYPE", 2.0f }, { "OSC2TYPE", 0.0f }, { "UNISON", 3.0f }, { "DETUNE", 0.03f },
                    { "OSC2SEMITONE", -12.0f }, { "OSC2MIX", 0.2f }, { "SUBOSCLEVEL", 0.14f }, { "NOISELEVEL", 0.06f },
                    { "RINGMOD", 0.0f }, { "FMAMOUNT", 60.0f }, { "SYNC", 0.04f }, { "FILTERTYPE", 3.0f },
                    { "FILTER2TYPE", 0.0f }, { "CUTOFF", 900.0f }, { "CUTOFF2", 2800.0f }, { "RESONANCE", 0.48f },
                    { "FILTERENVAMOUNT", 0.26f }, { "FILTERBALANCE", 0.32f }, { "AMPATTACK", 0.06f },
                    { "AMPDECAY", 0.52f }, { "AMPSUSTAIN", 0.58f }, { "AMPRELEASE", 0.44f }, { "FILTATTACK", 0.08f },
                    { "FILTDECAY", 0.46f }, { "FILTSUSTAIN", 0.18f }, { "FILTRELEASE", 0.38f }, { "LFO2RATE", 0.3f },
                    { "LFO2FILTER", 0.1f }, { "FXTYPE", 2.0f }, { "FXMIX", 0.18f }, { "FXINTENSITY", 0.38f },
                    { "DELAYSEND", 0.08f }, { "REVERBMIX", 0.14f },
                });
                break;
            case 19:
                applyAdvancedSettings ({
                    { "OSCTYPE", 1.0f }, { "OSC2TYPE", 1.0f }, { "UNISON", 6.0f }, { "DETUNE", 0.075f },
                    { "OSC2MIX", 0.44f }, { "SUBOSCLEVEL", 0.1f }, { "NOISELEVEL", 0.04f }, { "RINGMOD", 0.0f },
                    { "FILTER2TYPE", 3.0f }, { "CUTOFF", 1400.0f }, { "CUTOFF2", 5200.0f }, { "RESONANCE", 0.28f },
                    { "FILTERENVAMOUNT", 0.72f }, { "FILTERBALANCE", 0.38f }, { "AMPATTACK", 0.22f },
                    { "AMPDECAY", 1.0f }, { "AMPSUSTAIN", 0.78f }, { "AMPRELEASE", 2.0f }, { "FILTATTACK", 0.08f },
                    { "FILTDECAY", 2.2f }, { "FILTSUSTAIN", 0.12f }, { "FILTRELEASE", 1.4f }, { "LFO2RATE", 0.18f },
                    { "LFO2SHAPE", 1.0f }, { "LFO2FILTER", 0.22f }, { "FXTYPE", 4.0f }, { "FXMIX", 0.16f },
                    { "FXINTENSITY", 0.3f }, { "DELAYSEND", 0.12f }, { "DELAYTIME", 0.46f },
                    { "DELAYFEEDBACK", 0.34f }, { "REVERBMIX", 0.22f },
                });
                break;
            default:
                break;
        }

        remapLegacyFxType();
        syncLegacyLfoRouting();
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drumMachine)
    {
        setParameterActual ("DRUMMASTERLEVEL", 1.0f);
        setParameterActual ("DRUM_KICK_ATTACK", 0.72f);
        setParameterActual ("DRUM_SNARE_TONE", 0.56f);
        setParameterActual ("DRUM_SNARE_SNAPPY", 0.62f);

        for (size_t index = 0; index < drumVoiceTuneParamIds.size(); ++index)
            setParameterActual (drumVoiceTuneParamIds[index], drumMachineVoiceTuneDefaults[index]);

        for (size_t index = 0; index < drumVoiceDecayParamIds.size(); ++index)
            setParameterActual (drumVoiceDecayParamIds[index], drumMachineVoiceDecayDefaults[index]);

        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
            setParameterActual (drumVoiceLevelParamIds[index], drumMachineVoiceLevelDefaults[index]);

        switch (presetIndex)
        {
            case 1:
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("SYNC", 0.34f);
                setParameterActual ("FMAMOUNT", 120.0f);
                setParameterActual ("OSCGATE", 0.34f);
                setParameterActual ("CUTOFF", 10000.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                setParameterActual ("DRUM_KICK_ATTACK", 0.78f);
                setParameterActual ("DRUM_SNARE_SNAPPY", 0.58f);
                setParameterActual ("DRUMDECAY_CLOSED_HAT", 0.26f);
                setParameterActual ("DRUMDECAY_OPEN_HAT", 0.52f);
                break;
            case 2:
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("SYNC", 0.22f);
                setParameterActual ("FMAMOUNT", 90.0f);
                setParameterActual ("OSCGATE", 0.52f);
                setParameterActual ("CUTOFF", 7600.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.18f);
                setParameterActual ("DRUM_SNARE_TONE", 0.42f);
                setParameterActual ("DRUM_SNARE_SNAPPY", 0.78f);
                setParameterActual ("DRUMDECAY_KICK", 0.6f);
                setParameterActual ("DRUMDECAY_OPEN_HAT", 0.68f);
                break;
            default:
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("SYNC", 0.28f);
                setParameterActual ("FMAMOUNT", 150.0f);
                setParameterActual ("OSCGATE", 0.42f);
                setParameterActual ("CUTOFF", 14000.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.12f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::drum808)
    {
        setParameterActual ("DRUMMASTERLEVEL", 1.0f);
        setParameterActual ("DRUM_KICK_ATTACK", 0.34f);
        setParameterActual ("DRUM_SNARE_TONE", 0.48f);
        setParameterActual ("DRUM_SNARE_SNAPPY", 0.54f);

        for (size_t index = 0; index < drumVoiceTuneParamIds.size(); ++index)
            setParameterActual (drumVoiceTuneParamIds[index], drum808VoiceTuneDefaults[index]);

        for (size_t index = 0; index < drumVoiceDecayParamIds.size(); ++index)
            setParameterActual (drumVoiceDecayParamIds[index], drum808VoiceDecayDefaults[index]);

        for (size_t index = 0; index < drumVoiceLevelParamIds.size(); ++index)
            setParameterActual (drumVoiceLevelParamIds[index], drum808VoiceLevelDefaults[index]);

        switch (presetIndex)
        {
            case 1:
                setParameterActual ("DETUNE", -1.5f);
                setParameterActual ("SYNC", 0.06f);
                setParameterActual ("FMAMOUNT", 70.0f);
                setParameterActual ("OSCGATE", 1.35f);
                setParameterActual ("CUTOFF", 8400.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.04f);
                setParameterActual ("DRUMDECAY_KICK", 0.82f);
                setParameterActual ("DRUMDECAY_OPEN_HAT", 0.86f);
                setParameterActual ("DRUM_KICK_ATTACK", 0.26f);
                break;
            case 2:
                setParameterActual ("DETUNE", 0.8f);
                setParameterActual ("SYNC", 0.12f);
                setParameterActual ("FMAMOUNT", 120.0f);
                setParameterActual ("OSCGATE", 0.75f);
                setParameterActual ("CUTOFF", 11200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.08f);
                setParameterActual ("DRUM_SNARE_TONE", 0.58f);
                setParameterActual ("DRUM_SNARE_SNAPPY", 0.66f);
                setParameterActual ("DRUMTUNE_CRASH", 1.5f);
                break;
            default:
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("SYNC", 0.08f);
                setParameterActual ("FMAMOUNT", 90.0f);
                setParameterActual ("OSCGATE", 0.95f);
                setParameterActual ("CUTOFF", 9200.0f);
                setParameterActual ("FILTERENVAMOUNT", 0.05f);
                break;
        }
    }
    else if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        setParameterActual ("DRUMMASTERLEVEL", 1.0f);
        setParameterActual ("DETUNE", 0.0f);
        setParameterActual ("SYNC", 0.0f);
        setParameterActual ("FMAMOUNT", 0.0f);
        setParameterActual ("OSCGATE", 1.0f);
        setParameterActual ("FILTERTYPE", 0.0f);
        setParameterActual ("CUTOFF", 20000.0f);
        setParameterActual ("RESONANCE", 0.1f);
        setParameterActual ("FILTERENVAMOUNT", 0.0f);

        for (int padIndex = 0; padIndex < vecPadCount; ++padIndex)
        {
            setParameterActual (externalPadLevelParameterIdForIndex (padIndex).toRawUTF8(), 1.0f);
            setParameterActual (externalPadSustainParameterIdForIndex (padIndex).toRawUTF8(), 120.0f);
            setParameterActual (externalPadReleaseParameterIdForIndex (padIndex).toRawUTF8(), 0.2f);
            setParameterActual (externalPadSampleParameterIdForIndex (padIndex).toRawUTF8(), 0.0f);
        }
    }
    else
    {
        switch (presetIndex)
        {
            case 1:
                setParameterActual ("OSCTYPE", 2.0f);
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("FMAMOUNT", 0.0f);
                setParameterActual ("SYNC", 0.0f);
                setParameterActual ("CUTOFF", 1800.0f);
                setParameterActual ("RESONANCE", 0.16f);
                setParameterActual ("FILTERENVAMOUNT", 0.22f);
                break;
            case 2:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.05f);
                setParameterActual ("UNISON", 2.0f);
                setParameterActual ("CUTOFF", 3600.0f);
                setParameterActual ("AMPATTACK", 0.45f);
                setParameterActual ("AMPRELEASE", 1.2f);
                break;
            case 3:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("UNISON", 1.0f);
                setParameterActual ("CUTOFF", 2200.0f);
                setParameterActual ("AMPATTACK", 0.002f);
                setParameterActual ("AMPDECAY", 0.12f);
                setParameterActual ("AMPSUSTAIN", 0.7f);
                break;
            default:
                setParameterActual ("OSCTYPE", 1.0f);
                setParameterActual ("DETUNE", 0.0f);
                setParameterActual ("FMAMOUNT", 0.0f);
                setParameterActual ("SYNC", 0.0f);
                setParameterActual ("UNISON", 1.0f);
                setParameterActual ("CUTOFF", 2600.0f);
                setParameterActual ("RESONANCE", 0.18f);
                setParameterActual ("FILTERENVAMOUNT", 0.26f);
                break;
        }
    }

    if constexpr (buildFlavor() == InstrumentFlavor::advanced)
    {
        setParameterActual ("OSC3ENABLE", paramValue (apvts, "SUBOSCLEVEL") > 0.0001f ? 1.0f : 0.0f);
        setParameterActual ("RINGMODENABLE", paramValue (apvts, "RINGMOD") > 0.0001f ? 1.0f : 0.0f);
        setParameterActual ("FMENABLE", paramValue (apvts, "FMAMOUNT") > 0.0001f ? 1.0f : 0.0f);
        setParameterActual ("SYNCENABLE", paramValue (apvts, "SYNC") > 0.0001f ? 1.0f : 0.0f);
    }

    updateRenderParameters();
    if constexpr (buildFlavor() != InstrumentFlavor::vec1DrumPad)
        refreshSampleBank();
    else
        refreshExternalPadSamples();
    applyEnvelopeSettings();
}

void AdvancedVSTiAudioProcessor::parameterChanged (const juce::String& parameterID, float newValue)
{
    if (parameterID == "PRESET" && ! suppressPresetCallback)
    {
        currentProgramIndex = juce::jlimit (0, juce::jmax (0, getNumPrograms() - 1), juce::roundToInt (newValue));
        pendingPresetIndex.store (currentProgramIndex);
        triggerAsyncUpdate();
        return;
    }

    if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        if (parameterID.startsWith ("VECPADSELECT_"))
        {
            pendingExternalPadReload.store (true);
            triggerAsyncUpdate();
        }
    }
}

void AdvancedVSTiAudioProcessor::handleAsyncUpdate()
{
    const auto presetIndex = pendingPresetIndex.exchange (-1);
    const auto shouldRefreshExternalPads = pendingExternalPadReload.exchange (false);

    if (presetIndex >= 0)
        applyPresetByIndex (presetIndex);

    if constexpr (buildFlavor() == InstrumentFlavor::vec1DrumPad)
    {
        if (presetIndex >= 0 || shouldRefreshExternalPads)
            refreshExternalPadSamples();
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AdvancedVSTiAudioProcessor();
}
