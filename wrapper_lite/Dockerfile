# Builds wrapper-lite from source (Android NDK); the first build takes a while.
ARG WRAPPER_REF=56a7dcd726f2c2ef1d4420d6747ba0bafe215fee

FROM debian:13.2 AS build
ARG WRAPPER_REF
SHELL ["/bin/bash", "-c"]
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake unzip git wget ca-certificates lsb-release gnupg software-properties-common \
    && bash -c "$(wget -qO - https://apt.llvm.org/llvm.sh)" \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://dl.google.com/android/repository/android-ndk-r23b-linux.zip \
    && unzip -q android-ndk-r23b-linux.zip && rm android-ndk-r23b-linux.zip
RUN git init -q . && git remote add origin https://github.com/WorldObservationLog/wrapper.git \
    && git fetch -q --depth 1 origin "${WRAPPER_REF}" && git checkout -q FETCH_HEAD
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_HOST_LAUNCHERS=ON \
    && cmake --build build --target wrapper_lite_exe -j"$(nproc)" \
    && cmake --build build -j"$(nproc)"

FROM debian:13.2
RUN apt-get update && apt-get install -y --no-install-recommends jq && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/wrapper-lite /app/wrapper-lite
COPY --from=build /app/rootfs /app/rootfs
COPY run.sh /run.sh
RUN chmod +x /run.sh /app/wrapper-lite
CMD ["/run.sh"]
