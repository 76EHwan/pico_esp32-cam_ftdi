/*
 * TinyUSB device configuration: a single CDC-ACM port, nothing else.
 *
 * CFG_TUSB_MCU / CFG_TUSB_OS / CFG_TUSB_DEBUG are supplied on the command line
 * by the pico-sdk TinyUSB build (lib/tinyusb/hw/bsp/rp2040/family.cmake), so
 * they are only defaulted here for the benefit of editors and clangd.
 */
#ifndef _TUSB_CONFIG_H_
#define _TUSB_CONFIG_H_

#ifndef CFG_TUSB_MCU
#define CFG_TUSB_MCU            OPT_MCU_RP2040
#endif
#ifndef CFG_TUSB_OS
#define CFG_TUSB_OS             OPT_OS_PICO
#endif

#define CFG_TUSB_RHPORT0_MODE   (OPT_MODE_DEVICE | OPT_MODE_FULL_SPEED)

#ifndef CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_SECTION
#endif
#ifndef CFG_TUSB_MEM_ALIGN
#define CFG_TUSB_MEM_ALIGN      __attribute__((aligned(4)))
#endif

#define CFG_TUD_ENDPOINT0_SIZE  64

/* Two ports: 0 is the esptool bridge and the ESP32's own log, 1 carries the
 * vision stream and its command channel. Keeping them apart means a viewer can
 * stay attached while esptool reflashes the ESP32. */
#define CFG_TUD_CDC             2
#define CFG_TUD_MSC             0
#define CFG_TUD_HID             0
#define CFG_TUD_MIDI            0
#define CFG_TUD_VENDOR          0

/* esptool bursts a whole 1024-byte flash block at a time and expects the
 * adapter to keep taking bytes while the UART drains. 64 bytes of FIFO (the
 * TinyUSB default) makes it NAK constantly and roughly halves throughput. */
#define CFG_TUD_CDC_RX_BUFSIZE  1024
#define CFG_TUD_CDC_TX_BUFSIZE  1024
#define CFG_TUD_CDC_EP_BUFSIZE  64

#endif /* _TUSB_CONFIG_H_ */
