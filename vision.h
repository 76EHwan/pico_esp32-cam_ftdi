/*
 * The vision stage: pull 54x42 coverage frames out of the ESP32 byte stream,
 * turn them into a mask and a skeleton, and publish all three on CDC 1.
 */
#ifndef PICO_ESP32CAM_VISION_INCLUDED
#define PICO_ESP32CAM_VISION_INCLUDED

#include <stdbool.h>
#include <stdint.h>

#define VISION_W 54
#define VISION_H 42

/* Provided by vision.c */
void vision_init(void);

/* Feed one byte from the ESP32. Returns true if it belonged to a frame, in
 * which case the caller must not forward it as log text. */
bool vision_rx_byte(uint8_t b);

/* Drain the CDC 1 output and handle commands arriving on it. */
void vision_task(void);

/* True when no frame is part-way out to CDC 1, so log text may be interleaved. */
bool vision_out_idle(void);

/* Provided by pico_esp32-cam_ftdi.c: queue bytes for the ESP32's UART.
 * Returns how many were accepted. */
uint32_t esp_link_write(const uint8_t *p, uint32_t n);

/* The rate used whenever CDC 0 is closed, i.e. while the vision link is live. */
void     esp_link_set_vision_baud(uint32_t baud);
uint32_t esp_link_get_vision_baud(void);

#endif /* PICO_ESP32CAM_VISION_INCLUDED */
