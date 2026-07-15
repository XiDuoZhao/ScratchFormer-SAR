#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void write_png(const char *path, const unsigned char *data, int rgb) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    image.width = 256;
    image.height = 256;
    image.format = rgb ? PNG_FORMAT_RGB : PNG_FORMAT_GRAY;
    if (!png_image_write_to_file(&image, path, 0, data, 0, NULL)) {
        fprintf(stderr, "Cannot write %s: %s\n", path, image.message);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s SOURCE.tif OUTPUT_DIR AREA rgb|label\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int rgb = strcmp(argv[4], "rgb") == 0;
    TIFF *tif = TIFFOpen(argv[1], "r");
    if (tif == NULL) fail("Cannot open TIFF source");

    uint32_t width, height;
    uint16_t bits, samples;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
    if ((rgb && (bits != 8 || samples != 3)) || (!rgb && (bits != 1 || samples != 1))) {
        fail("Unexpected TIFF pixel format");
    }

    const uint32_t usable_width = width - width % 256;
    const uint32_t usable_height = height - height % 256;
    const tmsize_t tile_size = TIFFTileSize(tif);
    const tmsize_t tile_row_size = TIFFTileRowSize(tif);
    unsigned char *tile = _TIFFmalloc(tile_size);
    unsigned char *patch = malloc(256 * 256 * (rgb ? 3 : 1));
    if (tile == NULL || patch == NULL) fail("Out of memory");

    for (uint32_t top = 0; top < usable_height; top += 256) {
        for (uint32_t left = 0; left < usable_width; left += 256) {
            for (uint32_t dy = 0; dy < 256; dy += 128) {
                for (uint32_t dx = 0; dx < 256; dx += 128) {
                    if (TIFFReadTile(tif, tile, left + dx, top + dy, 0, 0) < 0) {
                        fail("Cannot decode TIFF tile");
                    }
                    for (uint32_t row = 0; row < 128; ++row) {
                        for (uint32_t col = 0; col < 128; ++col) {
                            const size_t target = ((dy + row) * 256 + dx + col) * (rgb ? 3 : 1);
                            if (rgb) {
                                const size_t source = row * tile_row_size + col * 3;
                                memcpy(patch + target, tile + source, 3);
                            } else {
                                const size_t source = row * tile_row_size + col / 8;
                                patch[target] = (tile[source] & (0x80 >> (col % 8))) ? 255 : 0;
                            }
                        }
                    }
                }
            }
            char path[4096];
            snprintf(path, sizeof(path), "%s/whu_%s_%05u_%05u.png", argv[2], argv[3], top, left);
            write_png(path, patch, rgb);
        }
    }
    free(patch);
    _TIFFfree(tile);
    TIFFClose(tif);
    return EXIT_SUCCESS;
}
