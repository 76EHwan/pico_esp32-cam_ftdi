/*
 * pico_esp32-cam_ftdi - an RP2040-Zero standing in for the FTDI adapter an
 * AI-Thinker ESP32-CAM would otherwise need.
 *
 * It enumerates as a USB CDC-ACM port, bridges that port to a PIO UART, follows
 * the host's requested baud rate, and reproduces the DTR/RTS reset circuit
 * esptool expects - so `esptool --port <this port> write_flash ...` and the
 * Arduino IDE both work with no buttons pressed and no IO0 jumper.
 *
 * The UART is on PIO rather than either hardware UART, so all four signal pins
 * below are free choices - change the defines and rewire, nothing else.
 *
 *   RP2040-Zero          ESP32-CAM (AI-Thinker)
 *   -----------          ----------------------
 *   5V          -------  5V
 *   GND         -------  GND
 *   GP0         -------  U0R (GPIO3)    PIO UART TX
 *   GP1         -------  U0T (GPIO1)    PIO UART RX
 *   GP2         -------  IO0            open drain: driven low, else Hi-Z
 *   GP3         -------  GND            jumper, only while flashing
 *
 * GP2 is open drain on purpose. IO0 doubles as the camera's XCLK once the ESP32
 * is running, so it must be released - not driven high - after flashing.
 *
 * EN is deliberately not wired: it is not on the AI-Thinker header and reaching
 * it means soldering to the RST button pad underneath. Reset is a button press
 * instead, and GP3 is the jumper that makes that possible - see below.
 */

#include <stdbool.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "pico/bootrom.h"
#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/sync.h"

#include "tusb.h"

#include "vision.h"

#include "pio_uart.pio.h"
#include "ws2812.pio.h"

// ---------------------------------------------------------------- pins / config

/* Any GPIO. The PIO UART has no pin mux constraints. */
#define PIN_UART_TX     0
#define PIN_UART_RX     1
#define PIN_ESP_IO0     2
#define PIN_FORCE_BOOT  3       /* jumper to GND, internally pulled up */
#define PIN_WS2812      16      /* on-board RGB LED, fixed by the board */

#define UART_PIO        pio0
#define LED_PIO         pio1

/* USB CDC interfaces */
#define CDC_ESP         0       /* esptool bridge and the ESP32's own log */
#define CDC_VISION      1       /* frame stream out, ASCII commands in    */

#define DEFAULT_BAUD    921600
#define MIN_BAUD        300
/* clkdiv is 16.8 fixed point over clk_sys at 8 cycles/bit, so the usable range
 * is clk_sys/8 down to clk_sys/(8*65536) - far wider than either end below. */
#define MAX_BAUD        6000000

// ---------------------------------------------------------------- USB -> UART ring

/* Single producer / single consumer, both in the main loop. head and tail
 * free-run and are masked on use, so count == head - tail needs no separate
 * "full" flag. */
#define TX_RB_BITS 11
#define TX_RB_SIZE (1u << TX_RB_BITS)
#define TX_RB_MASK (TX_RB_SIZE - 1u)

static struct {
    uint8_t buf[TX_RB_SIZE];
    uint32_t head, tail;
} tx_rb;

#define tx_count()     ((uint32_t)(tx_rb.head - tx_rb.tail))
#define tx_space()     (TX_RB_SIZE - tx_count())
#define tx_push(byte)  do { tx_rb.buf[tx_rb.head++ & TX_RB_MASK] = (byte); } while (0)
#define tx_pop()       (tx_rb.buf[tx_rb.tail++ & TX_RB_MASK])

// ---------------------------------------------------------------- UART -> USB ring

/* Written by DMA, drained by the main loop. 4 KiB is ~43 ms of slack at
 * 921600 baud; the alignment is what lets the DMA wrap the write address in
 * hardware, so the ring never needs servicing to stay coherent. */
#define RX_RING_BITS 12
#define RX_RING_SIZE (1u << RX_RING_BITS)
#define RX_RING_MASK (RX_RING_SIZE - 1u)

static uint8_t rx_ring[RX_RING_SIZE] __attribute__((aligned(RX_RING_SIZE)));
static uint32_t rx_tail;
static uint32_t rx_relap = RX_RING_SIZE;    /* source word for the restart channel */

static int dma_rx, dma_rx_restart;

static inline uint32_t rx_head(void) {
    uint32_t h = (uint32_t)(dma_hw->ch[dma_rx].write_addr - (uintptr_t)rx_ring);
    __dmb();    /* the pointer moved; make sure the byte behind it is visible */
    return h & RX_RING_MASK;
}

static inline uint32_t rx_count(void) {
    return (rx_head() - rx_tail) & RX_RING_MASK;
}

static inline uint8_t rx_pop(void) {
    uint8_t b = rx_ring[rx_tail];
    rx_tail = (rx_tail + 1) & RX_RING_MASK;
    return b;
}

// ---------------------------------------------------------------- PIO UART

static uint sm_tx, sm_rx, off_tx, off_rx;
static uint32_t current_baud = DEFAULT_BAUD;

static void esp_uart_dma_init(void) {
    dma_rx = dma_claim_unused_channel(true);
    dma_rx_restart = dma_claim_unused_channel(true);

    dma_channel_config c = dma_channel_get_default_config(dma_rx);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_8);
    channel_config_set_read_increment(&c, false);
    channel_config_set_write_increment(&c, true);
    channel_config_set_ring(&c, true, RX_RING_BITS);    /* wrap the write address */
    channel_config_set_dreq(&c, pio_get_dreq(UART_PIO, sm_rx, false));
    channel_config_set_chain_to(&c, dma_rx_restart);
    /* The program shifts right with an explicit PUSH, so the byte sits in bits
     * 31:24 of the FIFO word; read the top byte lane of RXF directly. A byte
     * read still pops the whole entry. */
    dma_channel_configure(dma_rx, &c, rx_ring,
                          (const volatile uint8_t *)&UART_PIO->rxf[sm_rx] + 3,
                          RX_RING_SIZE, false);

    /* When the data channel finishes a lap its write address has already
     * wrapped back to the start of the ring, so rearming it is a single write
     * of the transfer count to its trigger alias. Doing that from a chained
     * channel rather than an ISR means the ring never stops accepting bytes. */
    dma_channel_config r = dma_channel_get_default_config(dma_rx_restart);
    channel_config_set_transfer_data_size(&r, DMA_SIZE_32);
    channel_config_set_read_increment(&r, false);
    channel_config_set_write_increment(&r, false);
    dma_channel_configure(dma_rx_restart, &r,
                          &dma_hw->ch[dma_rx].al1_transfer_count_trig,
                          &rx_relap, 1, false);

    dma_channel_start(dma_rx);
}

static void esp_uart_init(void) {
    off_tx = pio_add_program(UART_PIO, &uart_tx_program);
    off_rx = pio_add_program(UART_PIO, &uart_rx_program);
    sm_tx = (uint)pio_claim_unused_sm(UART_PIO, true);
    sm_rx = (uint)pio_claim_unused_sm(UART_PIO, true);

    uart_tx_program_init(UART_PIO, sm_tx, off_tx, PIN_UART_TX, DEFAULT_BAUD);
    uart_rx_program_init(UART_PIO, sm_rx, off_rx, PIN_UART_RX, DEFAULT_BAUD);

    esp_uart_dma_init();
}

static void esp_uart_set_baud(uint32_t baud) {
    float div = (float)clock_get_hz(clk_sys) / (8.0f * (float)baud);
    current_baud = baud;

    pio_sm_set_clkdiv(UART_PIO, sm_tx, div);
    pio_sm_clkdiv_restart(UART_PIO, sm_tx);

    /* RX has to be restarted outright: a byte half-received at the old rate
     * would otherwise be finished at the new one and pushed as garbage.
     * Clearing the FIFO simply deasserts DREQ, so the DMA stays armed. */
    pio_sm_set_enabled(UART_PIO, sm_rx, false);
    pio_sm_set_clkdiv(UART_PIO, sm_rx, div);
    pio_sm_clear_fifos(UART_PIO, sm_rx);
    pio_sm_restart(UART_PIO, sm_rx);
    pio_sm_clkdiv_restart(UART_PIO, sm_rx);
    pio_sm_exec(UART_PIO, sm_rx, pio_encode_jmp(off_rx));
    pio_sm_set_enabled(UART_PIO, sm_rx, true);
}

static void esp_uart_tx_pump(void) {
    while (tx_count() && !pio_sm_is_tx_fifo_full(UART_PIO, sm_tx)) {
        pio_sm_put(UART_PIO, sm_tx, tx_pop());
    }
}

static void esp_uart_tx_drain(void) {
    while (tx_count()) esp_uart_tx_pump();
    while (!pio_sm_is_tx_fifo_empty(UART_PIO, sm_tx)) tight_loop_contents();
    /* The FIFO is empty but the last frame is still being shifted out; PIO has
     * no "transmitter idle" flag, so wait out one 10-bit frame. */
    busy_wait_us(1 + (10u * 1000000u) / current_baud);
}

// ---------------------------------------------------------------- control lines

static bool io0_low;
static bool host_dtr, host_rts, force_boot;

/* Drive low or release. The ESP32 pulls both lines up on board. */
static void od_set(uint gpio, bool low) {
    gpio_set_dir(gpio, low ? GPIO_OUT : GPIO_IN);
}

static void od_init(uint gpio) {
    gpio_init(gpio);
    gpio_put(gpio, 0);              /* output register stays 0 forever */
    gpio_set_dir(gpio, GPIO_IN);    /* start released */
    gpio_disable_pulls(gpio);
}

/*
 * esptool's classic reset, per its own reset.py:
 *
 *     RTS asserted -> EN  low   (chip held in reset)
 *     DTR asserted -> IO0 low   (boot into the serial bootloader)
 *
 * EN is not wired here, so only the DTR half does anything - but the real
 * two-transistor circuit cross-couples the pair so that asserting *both* drives
 * neither, and that half matters on its own. Windows and many Linux drivers
 * raise DTR and RTS together the moment a port is opened; without the
 * cross-coupling, opening a terminal would hold IO0 low, and IO0 is the
 * camera's XCLK.
 */
static void control_update(void) {
    io0_low = force_boot || ((host_dtr != host_rts) && host_dtr);
    od_set(PIN_ESP_IO0, io0_low);
}

/*
 * Jumper GP3 to GND to hold IO0 low for as long as it is in.
 *
 * With EN unwired the ESP32 is reset by hand, and the rule above releases IO0
 * as soon as the host has both DTR and RTS asserted. esptool only drives IO0
 * low for ~50 ms inside its reset sequence - far too narrow to hit with a
 * button press. With the jumper in, press the module's own RST button and it
 * comes up in the bootloader.
 *
 * Take the jumper out before the final reset: IO0 is the camera's XCLK.
 */
static void force_boot_task(void) {
    bool now = !gpio_get(PIN_FORCE_BOOT);
    if (now != force_boot) {
        force_boot = now;
        control_update();
    }
}

// ---------------------------------------------------------------- CDC callbacks

/* Last rate each port was opened at, for the 1200 bps touch below. */
static uint32_t touch_baud[2];

/* Only CDC 0 drives the ESP32. A viewer opening the vision port asserts DTR and
 * RTS exactly like any other host would, and must not reset the camera. */
void tud_cdc_line_state_cb(uint8_t itf, bool dtr, bool rts) {
    /* Arduino's 1200 bps touch: open a port at 1200 baud, then drop DTR, and
     * the board reboots into its own bootloader. Owning USB directly means
     * pico_stdio_usb's reset interface is not here, so without this every
     * reflash of this board means reaching for the BOOT button - and the point
     * of putting the tuning on this side was that reflashing it is cheap.
     * Nothing on this link runs at 1200 baud, so it cannot fire by accident. */
    if (!dtr && itf < 2 && touch_baud[itf] == 1200) {
        reset_usb_boot(0, 0);
    }

    if (itf != CDC_ESP) return;
    host_dtr = dtr;
    host_rts = rts;
    control_update();
}

/*
 * Two things want to own the UART rate. esptool has to be able to move it (it
 * negotiates up to 460800 or 921600 mid-session), but the vision link needs a
 * fixed rate that survives nobody having the bridge port open at all - and with
 * the rate driven purely by CDC 0's line coding it would sit at whatever the
 * last host asked for, or the power-on default, and every frame would be shred.
 *
 * So: CDC 0's line coding applies only while CDC 0 is actually open, and the
 * vision rate is restored the moment it closes.
 */
static uint32_t vision_baud = DEFAULT_BAUD;
static uint32_t bridge_baud = DEFAULT_BAUD;
static bool bridge_active;
static absolute_time_t bridge_deadline;

/*
 * "Is the bridge port in use" cannot be answered with tud_cdc_n_connected():
 * that is DTR, and esptool's reset sequence ends with DTR deasserted and stays
 * that way for the whole session (its reset.py literally finishes on
 * _setDTR(False)). Reading that as "port closed" reverted the UART to the
 * vision rate while the ROM bootloader sat at 115200, and stopped forwarding
 * the ESP32's replies - esptool saw silence and gave up.
 *
 * So treat it as an activity window instead: the host setting a line coding or
 * sending a byte marks the bridge in use, and it lapses after a few quiet
 * seconds. esptool sends continuously, so it holds the window open by itself.
 */
static void bridge_touch(void) {
    bridge_deadline = make_timeout_time_ms(3000);
    if (!bridge_active) {
        bridge_active = true;
        esp_uart_set_baud(bridge_baud);
        vision_init();          /* drop any half-parsed frame */
    }
}

static void link_baud_task(void) {
    if (!bridge_active) return;
    if (absolute_time_diff_us(get_absolute_time(), bridge_deadline) > 0) return;
    bridge_active = false;
    esp_uart_set_baud(vision_baud);
    vision_init();
}

void esp_link_set_vision_baud(uint32_t baud) {
    if (baud < MIN_BAUD || baud > MAX_BAUD) return;
    vision_baud = baud;
    if (!bridge_active) esp_uart_set_baud(vision_baud);
}

uint32_t esp_link_get_vision_baud(void) { return vision_baud; }

uint32_t esp_link_write(const uint8_t *p, uint32_t n) {
    uint32_t i = 0;
    while (i < n && tx_space()) tx_push(p[i++]);
    return i;
}

void tud_cdc_line_coding_cb(uint8_t itf, cdc_line_coding_t const *lc) {
    if (itf < 2) touch_baud[itf] = lc->bit_rate;

    if (itf != CDC_ESP) return;
    if (lc->bit_rate < MIN_BAUD || lc->bit_rate > MAX_BAUD) return;
    bridge_baud = lc->bit_rate;

    /* Finish whatever is already queued at the old rate - esptool switches
     * baud right after a command it expects to have gone out in full. */
    esp_uart_tx_drain();
    if (bridge_active) esp_uart_set_baud(bridge_baud);
    bridge_touch();     /* applies it on the first call of a session */
    /* Data bits, parity and stop bits are fixed at 8N1: that is all esptool and
     * the ESP32 ROM bootloader ever use, and encoding the rest would cost PIO
     * instructions for no gain. */
}

// ---------------------------------------------------------------- bridge

static uint32_t activity;       /* bytes moved since the last LED tick */

/*
 * The ESP32's own chatter. Best effort on purpose - dropping a log line beats
 * stalling the wire - and deliberately not gated on tud_cdc_n_connected(),
 * which is DTR and which esptool leaves deasserted.
 *
 * The mirror to CDC 1 waits for the vision stage to be between frames, so a
 * log line cannot land in the middle of one and fail its CRC.
 */
static void log_out(const uint8_t *p, uint32_t n) {
    if (tud_cdc_n_write_available(CDC_ESP) >= n) {
        tud_cdc_n_write(CDC_ESP, p, n);
        tud_cdc_n_write_flush(CDC_ESP);
    }
    if (vision_out_idle() && tud_cdc_n_write_available(CDC_VISION) >= n) {
        tud_cdc_n_write(CDC_VISION, p, n);
        tud_cdc_n_write_flush(CDC_VISION);
    }
}

static void bridge_task(void) {
    uint8_t buf[64];

    /* USB -> UART. Only pull from the CDC FIFO while the ring can take a full
     * packet; leaving the rest in the FIFO is what applies backpressure to the
     * host, which is the only flow control this link has. */
    while (tud_cdc_n_available(CDC_ESP) && tx_space() >= sizeof buf) {
        uint32_t n = tud_cdc_n_read(CDC_ESP, buf, sizeof buf);
        for (uint32_t i = 0; i < n; i++) tx_push(buf[i]);
        activity += n;
        bridge_touch();
    }
    esp_uart_tx_pump();

    if (bridge_active) {
        /*
         * esptool owns the wire. Pass it through byte for byte, losing none:
         * its SLIP packets are binary and can hold a 0xA5, which the frame
         * parser would swallow, and a reply that arrives with a hole in it
         * kills the session.
         */
        while (rx_count()) {
            uint32_t room = tud_cdc_n_write_available(CDC_ESP);
            if (!room) break;               /* host is behind; try again later */
            uint32_t n = rx_count();
            if (n > room) n = room;
            if (n > sizeof buf) n = sizeof buf;
            for (uint32_t i = 0; i < n; i++) buf[i] = rx_pop();
            tud_cdc_n_write(CDC_ESP, buf, n);
            activity += n;
        }
        tud_cdc_n_write_flush(CDC_ESP);
        return;
    }

    /*
     * Frames to the vision stage, everything else out as log text. Frames are
     * binary and log output is ASCII, and ASCII never reaches 0x80, so the two
     * share one wire without needing to be separated first.
     */
    uint32_t n = 0;
    while (rx_count()) {
        uint8_t b = rx_pop();
        activity++;
        if (vision_rx_byte(b)) continue;
        buf[n++] = b;
        if (n == sizeof buf) { log_out(buf, n); n = 0; }
    }
    if (n) log_out(buf, n);
}

// ---------------------------------------------------------------- status LED

static uint led_sm;

static void led_put(uint32_t rgb) {
    uint32_t grb = ((rgb & 0x00FF00u) << 8) | ((rgb & 0xFF0000u) >> 8) | (rgb & 0x0000FFu);
    pio_sm_put_blocking(LED_PIO, led_sm, grb << 8u);
}

static void led_init(void) {
    uint offset = pio_add_program(LED_PIO, &ws2812_program);
    led_sm = (uint)pio_claim_unused_sm(LED_PIO, true);
    ws2812_program_init(LED_PIO, led_sm, offset, PIN_WS2812, 800000);
}

/* Kept dim - the RP2040-Zero's LED is uncomfortably bright at full scale. */
static void led_task(void) {
    static absolute_time_t next;
    static uint32_t shown = 0xFFFFFFFFu;

    if (absolute_time_diff_us(get_absolute_time(), next) > 0) return;
    next = make_timeout_time_ms(40);

    uint32_t c;
    if (force_boot)                 c = 0x200020;   /* magenta - boot jumper in */
    else if (!tud_mounted())        c = 0x080000;   /* red     - no host        */
    else if (io0_low)               c = 0x100010;   /* magenta - esptool on IO0 */
    else if (activity)              c = 0x001010;   /* cyan    - data moving    */
    else if (tud_cdc_n_connected(CDC_ESP))
                                    c = 0x000800;   /* green   - port open      */
    else                            c = 0x000004;   /* blue    - idle           */
    activity = 0;

    if (c != shown) {
        shown = c;
        led_put(c);
    }
}

// ---------------------------------------------------------------- main

int main(void) {
    od_init(PIN_ESP_IO0);
    gpio_init(PIN_FORCE_BOOT);
    gpio_set_dir(PIN_FORCE_BOOT, GPIO_IN);
    gpio_pull_up(PIN_FORCE_BOOT);
    force_boot_task();          /* honour the jumper before USB comes up */
    esp_uart_init();
    led_init();
    led_put(0x080000);
    vision_init();

    tusb_init();

    for (;;) {
        tud_task();
        link_baud_task();
        bridge_task();
        vision_task();
        force_boot_task();
        led_task();
    }
}
