; Build with Inno Setup 6.7+ from the source directory.
#ifndef AppBuild
  #define AppBuild "dist\LumenARW"
#endif

[Setup]
AppId={{F1CA8FF7-EB54-4B53-81E3-4CC183C8C1B9}
AppName=Lumen ARW
AppVersion=1.2.2
AppPublisher=Lumen ARW
DefaultDirName={localappdata}\Programs\Lumen ARW
DefaultGroupName=Lumen ARW
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19045
OutputDir=.publish\v122\packages
OutputBaseFilename=LumenARW-1.2.2-Setup
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupIconFile=assets\lumen.ico
UninstallDisplayIcon={app}\LumenARW.exe
LicenseFile=LICENSE
CloseApplications=yes
RestartApplications=no
Uninstallable=not PortableMode
CreateUninstallRegKey=not PortableMode
UsePreviousAppDir=not PortableMode

[Languages]
Name: "zhcn"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式"; Flags: unchecked; Check: not PortableMode

[Files]
Source: "{#AppBuild}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Lumen ARW"; Filename: "{app}\LumenARW.exe"; Check: not PortableMode
Name: "{userdesktop}\Lumen ARW"; Filename: "{app}\LumenARW.exe"; Tasks: desktopicon; Check: not PortableMode

[Run]
Filename: "{app}\LumenARW.exe"; Description: "启动 Lumen ARW"; Flags: nowait postinstall skipifsilent

[Code]
function PortableMode: Boolean;
begin
  Result := ExpandConstant('{param:PORTABLE|0}') = '1';
end;
