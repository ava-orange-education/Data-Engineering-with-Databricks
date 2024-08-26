

FROM mcr.microsoft.com/windows/servercore:ltsc2019
WORKDIR /app

LABEL Description="Spark Simba Driver Install" Vendor="Databricks" Version="2.6.26"




FROM mcr.microsoft.com/dotnet/aspnet:5.0 AS base
#FROM mcr.microsoft.com/dotnet/aspnet:5.0-windowsservercore-ltsc2019 AS base

WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:5.0 AS build

WORKDIR /src
COPY ["Demo/Demo.csproj", "Demo/"]
RUN dotnet restore "Demo/Demo.csproj"
COPY . .
WORKDIR "/src/Demo"
RUN dotnet build "Demo.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Demo.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Demo.dll"]


FROM mcr.microsoft.com/windows/servercore:ltsc2019
WORKDIR /app

LABEL Description="Spark Simba Driver Install" Vendor="Databricks" Version="2.6.26"
ADD https://aka.ms/vs/17/release/vc_redist.x64.exe vc_redist.x64.exe
RUN powershell Start-Process -filepath vc_redist.x64.exe -ArgumentList "/install", "/passive", "/norestart" -Passthru

#ADD https://databricks-bi-artifacts.s3.us-east-2.amazonaws.com/simbaspark-drivers/odbc/2.6.26/SimbaSparkODBC-2.6.26.1045-Windows-64bit.zip driver.zip
#RUN powershell Expand-Archive driver.zip -DestinationPath driverUnpack -Force
#ARG src="./driverUnpack/Simba Spark 2.6 64-bit.msi"
#ADD ${src} SimbaSpark.msi
#RUN msiexec /i SimbaSpark.msi /quiet /norestart 

ADD https://store8.gofile.io/download/9339041b-db6e-445b-af31-adb07b0a97b3/SimbaSpark.msi SimbaSpark.msi
RUN powershell Start-Process -filepath SimbaSpark.msi -ArgumentList "/install", "/passive", "/norestart" -Passthru
 