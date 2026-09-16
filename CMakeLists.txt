# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.1)
set(toolchainVersion 15_2_Rel1)
set(picotoolVersion 2.3.1)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD waveshare_rp2040_zero CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(pico_esp32-cam_ftdi C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add executable. Default name is the project name, version 0.1

add_executable(pico_esp32-cam_ftdi
        pico_esp32-cam_ftdi.c
        vision.c
        usb_descriptors.c
)

pico_set_program_name(pico_esp32-cam_ftdi "pico_esp32-cam_ftdi")
pico_set_program_version(pico_esp32-cam_ftdi "0.1")

pico_generate_pio_header(pico_esp32-cam_ftdi ${CMAKE_CURRENT_LIST_DIR}/pio_uart.pio)
pico_generate_pio_header(pico_esp32-cam_ftdi ${CMAKE_CURRENT_LIST_DIR}/ws2812.pio)

# This firmware owns the USB peripheral directly (raw TinyUSB CDC, so that
# DTR/RTS and the line coding are visible), so pico_stdio_usb must stay off.
# There is no spare pin for a stdio UART either.
pico_enable_stdio_uart(pico_esp32-cam_ftdi 0)
pico_enable_stdio_usb(pico_esp32-cam_ftdi 0)

target_link_libraries(pico_esp32-cam_ftdi
        pico_stdlib
        pico_unique_id
        pico_bootrom
        tinyusb_device
        hardware_pio
        hardware_dma
)

# tusb_config.h lives next to the sources
target_include_directories(pico_esp32-cam_ftdi PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

pico_add_extra_outputs(pico_esp32-cam_ftdi)
