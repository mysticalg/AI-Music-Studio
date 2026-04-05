#define MyAppName "Mutagen"
#ifndef MyAppVersion
  #define MyAppVersion "dev"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "Mysticalg"
#endif
#ifndef MyAppURL
  #define MyAppURL "https://mysticalg.github.io/AI-Music-Studio/"
#endif
#ifndef MyAppExeName
  #define MyAppExeName "Mutagen.exe"
#endif
#ifndef SourceRoot
  #error SourceRoot must be defined on the command line.
#endif
#ifndef OutputRoot
  #error OutputRoot must be defined on the command line.
#endif

[Setup]
AppId={{7C8459F9-5C5F-40D4-8D67-7B14A409CB31}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf64}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
OutputDir={#OutputRoot}
OutputBaseFilename=Mutagen-{#MyAppVersion}-setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full installation"
Name: "minimal"; Description: "Minimal installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "main"; Description: "Mutagen application"; Types: full minimal custom; Flags: fixed
Name: "bundledvsti"; Description: "Bundled native VST instruments and sample libraries"; Types: full custom

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut for Mutagen"; GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "installacestep"; Description: "Install ACE-Step local audio generation backend (downloads extra files)"; GroupDescription: "Optional components:"; Flags: unchecked

[Files]
Source: "{#SourceRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "vsti\*"
Source: "{#SourceRoot}\vsti\*"; DestDir: "{app}\vsti"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: bundledvsti

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoLogo -NoProfile -ExecutionPolicy Bypass -File ""{app}\support\install_ace_step_backend.ps1"""; StatusMsg: "Installing ACE-Step local audio generation backend..."; Flags: runascurrentuser waituntilterminated skipifsilent; Tasks: installacestep
Filename: "{app}\{#MyAppExeName}"; Description: "Run {#MyAppName} now"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
