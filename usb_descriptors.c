/* USB descriptors: one CDC-ACM interface, presented as a plain serial port. */

#include <string.h>

#include "pico/unique_id.h"
#include "tusb.h"

/* pid.codes 0x1209:0x0001 is the block explicitly set aside for prototypes.
 * Windows 10/11 bind usbccgp + usbser to an IAD composite CDC by class, with no
 * .inf and regardless of the IDs, and esptool does not look at them at all. */
#define USB_VID 0x1209
#define USB_PID 0x0001
#define USB_BCD 0x0200

static tusb_desc_device_t const desc_device = {
    .bLength         = sizeof(tusb_desc_device_t),
    .bDescriptorType = TUSB_DESC_DEVICE,
    .bcdUSB          = USB_BCD,

    /* IAD, so the two CDC interfaces are seen as one function. */
    .bDeviceClass    = TUSB_CLASS_MISC,
    .bDeviceSubClass = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol = MISC_PROTOCOL_IAD,

    .bMaxPacketSize0 = CFG_TUD_ENDPOINT0_SIZE,

    .idVendor        = USB_VID,
    .idProduct       = USB_PID,
    .bcdDevice       = 0x0100,

    .iManufacturer   = 0x01,
    .iProduct        = 0x02,
    .iSerialNumber   = 0x03,

    .bNumConfigurations = 0x01,
};

uint8_t const *tud_descriptor_device_cb(void) {
    return (uint8_t const *)&desc_device;
}

enum {
    ITF_NUM_CDC0 = 0,       /* esptool bridge / ESP32 log */
    ITF_NUM_CDC0_DATA,
    ITF_NUM_CDC1,           /* vision stream + commands  */
    ITF_NUM_CDC1_DATA,
    ITF_NUM_TOTAL
};

#define EPNUM_CDC0_NOTIF 0x81
#define EPNUM_CDC0_OUT   0x02
#define EPNUM_CDC0_IN    0x82
#define EPNUM_CDC1_NOTIF 0x83
#define EPNUM_CDC1_OUT   0x04
#define EPNUM_CDC1_IN    0x84

#define CONFIG_TOTAL_LEN (TUD_CONFIG_DESC_LEN + 2 * TUD_CDC_DESC_LEN)

static uint8_t const desc_configuration[] = {
    TUD_CONFIG_DESCRIPTOR(1, ITF_NUM_TOTAL, 0, CONFIG_TOTAL_LEN, 0x00, 100),
    TUD_CDC_DESCRIPTOR(ITF_NUM_CDC0, 4, EPNUM_CDC0_NOTIF, 8, EPNUM_CDC0_OUT, EPNUM_CDC0_IN, 64),
    TUD_CDC_DESCRIPTOR(ITF_NUM_CDC1, 5, EPNUM_CDC1_NOTIF, 8, EPNUM_CDC1_OUT, EPNUM_CDC1_IN, 64),
};

uint8_t const *tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return desc_configuration;
}

enum { STRID_LANGID = 0, STRID_MANUFACTURER, STRID_PRODUCT, STRID_SERIAL, STRID_CDC0, STRID_CDC1 };

static char const *const string_desc_arr[] = {
    (const char[]){0x09, 0x04},   /* 0: English (0x0409) */
    "Waveshare",                  /* 1: manufacturer     */
    "RP2040-Zero ESP32-CAM Link", /* 2: product          */
    NULL,                         /* 3: serial, from the flash unique ID */
    "ESP32-CAM UART",             /* 4: CDC 0 - esptool bridge and log */
    "Vision Stream",              /* 5: CDC 1 - frames and commands    */
};

static uint16_t _desc_str[32 + 1];

uint16_t const *tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    (void)langid;
    size_t chr_count;

    if (index == STRID_LANGID) {
        memcpy(&_desc_str[1], string_desc_arr[0], 2);
        chr_count = 1;
    } else if (index == STRID_SERIAL) {
        char id[2 * PICO_UNIQUE_BOARD_ID_SIZE_BYTES + 1];
        pico_get_unique_board_id_string(id, sizeof id);
        chr_count = strlen(id);
        for (size_t i = 0; i < chr_count; i++) _desc_str[1 + i] = (uint16_t)id[i];
    } else {
        if (index >= TU_ARRAY_SIZE(string_desc_arr)) return NULL;
        char const *str = string_desc_arr[index];
        chr_count = strlen(str);
        size_t const max_count = TU_ARRAY_SIZE(_desc_str) - 1;
        if (chr_count > max_count) chr_count = max_count;
        for (size_t i = 0; i < chr_count; i++) _desc_str[1 + i] = (uint16_t)str[i];
    }

    _desc_str[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * chr_count + 2));
    return _desc_str;
}
