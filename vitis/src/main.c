/*
 * CNN NPU (Neural Processing Unit) Implementation
 *
 * This code implements a Convolutional Neural Network that can run on:
 * - Linux systems (using memory mapping)
 * - Windows systems (for development/testing)
 * - Xilinx bare-metal systems (Ultra96, Zynq, etc.)
 *
 * The CNN architecture implements:
 * 1. Convolution layer (5x5 kernels, 4 filters)
 * 2. Max pooling layer (2x2)
 * 3. Fully connected layer (10 output classes)
 *
 * Designed for MNIST-like 28x28 grayscale image classification
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "xil_printf.h"  // Xilinx printf implementation
#include "xil_io.h"      // Xilinx I/O functions
#include "xparameters.h" // Xilinx hardware parameters

// Platform-specific includes for sleep functionality
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h> // For Sleep() on Windows
#elif defined(__linux__)
#include <unistd.h> // For usleep() on Linux
#else
// For bare-metal Xilinx platforms
// Note: If usleep_A9 was defined in removed "sleep.h",
// it would need to be declared here for Xilinx ARM platforms
#endif

// Memory mapping includes - Linux vs bare-metal
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#define USE_LINUX_MMAP // Flag to use Linux memory mapping
#else
// For bare-metal Xilinx systems
#include "xil_mmu.h"   // Memory Management Unit functions
#include "xil_cache.h" // Cache management functions
#endif

//=============================================================================
// NPU Hardware Register Definitions
//=============================================================================

// NPU base address - either from hardware parameters or default
#ifndef XPAR_NPU_0_S00_AXI_BASEADDR
#define NPU_BASE_ADDR 0xA0000000 // Default base address
#else
#define NPU_BASE_ADDR XPAR_NPU_0_S00_AXI_BASEADDR
#endif

// NPU register offsets from base address
#define NPU_CONTROL_REG 0x00     // Control register - start/stop operations
#define NPU_STATUS_REG 0x04      // Status register - operation status/completion
#define NPU_INPUT_ADDR_REG 0x08  // Input data memory address
#define NPU_OUTPUT_ADDR_REG 0x0C // Output data memory address
#define NPU_WEIGHT_ADDR_REG 0x10 // Weight/kernel memory address
#define NPU_CONFIG_REG 0x14      // Configuration - layer parameters

// Memory mapping constants
#define MAP_SIZE 4096UL         // 4KB memory map size
#define MAP_MASK (MAP_SIZE - 1) // Mask for address alignment

//=============================================================================
// CNN Architecture Parameters
//=============================================================================

#define INPUT_SIZE 28   // 28x28 input images (MNIST standard)
#define CONV1_FILTERS 4 // Number of convolution filters in first layer
#define CONV1_KERNEL 5  // 5x5 convolution kernel size
#define POOL1_SIZE 2    // 2x2 max pooling window
#define FC_NEURONS 10   // Output neurons (10 classes for digit recognition)

//=============================================================================
// Data Type Definitions
//=============================================================================

typedef int16_t data_t; // 16-bit signed integer for feature data
typedef int32_t acc_t;  // 32-bit signed integer for accumulation

//=============================================================================
// Memory Access Pointers
//=============================================================================

#ifdef USE_LINUX_MMAP
                       // Linux: Use memory mapping for NPU register access
volatile uint32_t *npu_mem = NULL; // Pointer to mapped NPU registers
static int mem_fd = -1;            // File descriptor for /dev/mem
#else
                       // Bare-metal: Direct memory access to NPU registers
#define npu_mem ((volatile uint32_t *)NPU_BASE_ADDR)
#endif

//=============================================================================
// CNN Weights and Biases (Pre-trained/Initialized)
//=============================================================================

// First convolution layer weights [filter][row][col]
// These are simple edge detection and pattern recognition filters
data_t conv1_weights[CONV1_FILTERS][CONV1_KERNEL][CONV1_KERNEL] = {
    // Filter 0 - Edge detector (center-surround pattern)
    {{-1, -1, -1, -1, -1},
     {-1, 2, 2, 2, -1},
     {-1, 2, 8, 2, -1}, // Strong center response
     {-1, 2, 2, 2, -1},
     {-1, -1, -1, -1, -1}},

    // Filter 1 - Horizontal line detector
    {{-1, -1, -1, -1, -1},
     {2, 2, 2, 2, 2}, // Horizontal edge response
     {2, 2, 2, 2, 2},
     {2, 2, 2, 2, 2},
     {-1, -1, -1, -1, -1}},

    // Filter 2 - Vertical line detector
    {{-1, 2, 2, 2, -1},
     {-1, 2, 2, 2, -1}, // Vertical edge response
     {-1, 2, 2, 2, -1},
     {-1, 2, 2, 2, -1},
     {-1, 2, 2, 2, -1}},

    // Filter 3 - Diagonal line detector
    {{4, -1, -1, -1, -1},
     {-1, 4, -1, -1, -1}, // Diagonal edge response
     {-1, -1, 4, -1, -1},
     {-1, -1, -1, 4, -1},
     {-1, -1, -1, -1, 4}}};

// Bias values for each convolution filter
data_t conv1_bias[CONV1_FILTERS] = {10, 5, 5, 8};

// Fully connected layer weights [output_neuron][input_features]
// Input features = 12x12x4 = 576 (after conv+pool layers)
data_t fc_weights[FC_NEURONS][12 * 12 * CONV1_FILTERS];

// Bias values for each output neuron (representing digits 0-9)
data_t fc_bias[FC_NEURONS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

//=============================================================================
// Function Prototypes
//=============================================================================

// NPU hardware interface functions
int init_npu(void);                                 // Initialize NPU hardware
void cleanup_npu(void);                             // Cleanup NPU resources
int write_npu_reg(uint32_t offset, uint32_t value); // Write to NPU register
uint32_t read_npu_reg(uint32_t offset);             // Read from NPU register

// CNN processing functions
void generate_test_image(data_t *image);                                       // Create test input image
void print_feature_map(data_t *data, int height, int width, const char *name); // Debug output
int run_cnn_inference(data_t *input_image, int *output_class);                 // Main CNN processing
void software_relu(data_t *data, int size);                                    // ReLU activation function
int argmax(data_t *data, int size);                                            // Find maximum value index

// Utility functions
void test_npu_registers(void);  // Hardware register test
uint32_t get_timer_value(void); // Performance timing

//=============================================================================
// Static Memory Buffers (Alternative to malloc for embedded systems)
//=============================================================================

// Calculate buffer sizes at compile time
#define CONV1_OUT_WIDTH (INPUT_SIZE - CONV1_KERNEL + 1)                            // 28-5+1 = 24
#define CONV1_OUT_HEIGHT (INPUT_SIZE - CONV1_KERNEL + 1)                           // 28-5+1 = 24
#define CONV1_OUTPUT_ELEMENTS (CONV1_OUT_WIDTH * CONV1_OUT_HEIGHT * CONV1_FILTERS) // 24*24*4 = 2304

#define POOL1_OUT_WIDTH (CONV1_OUT_WIDTH / POOL1_SIZE)                             // 24/2 = 12
#define POOL1_OUT_HEIGHT (CONV1_OUT_HEIGHT / POOL1_SIZE)                           // 24/2 = 12
#define POOL1_OUTPUT_ELEMENTS (POOL1_OUT_WIDTH * POOL1_OUT_HEIGHT * CONV1_FILTERS) // 12*12*4 = 576

#define FC_OUTPUT_ELEMENTS (FC_NEURONS) // 10

// Static buffer for convolution output (avoids malloc in embedded systems)
static data_t conv1_output_static_buffer[CONV1_OUTPUT_ELEMENTS];

// Uncomment these if malloc consistently fails on your target system:
// static data_t pool1_output_static_buffer[POOL1_OUTPUT_ELEMENTS];
// static data_t fc_output_static_buffer[FC_OUTPUT_ELEMENTS];

//=============================================================================
// Main Function
//=============================================================================

int main()
{
    xil_printf("CNN NPU Test Starting...\r\n");

    // Initialize NPU hardware
    if (init_npu() != 0)
    {
        xil_printf("Failed to initialize NPU\r\n");
        return -1;
    }

    // Allocate memory for input image (28x28x2 bytes = 1568 bytes)
    data_t *input_image = (data_t *)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(data_t));
    if (input_image == NULL)
    {
        xil_printf("Failed to allocate memory for input image\r\n");
        cleanup_npu();
        return -1;
    }

    // Generate a test image (simulated handwritten digit)
    generate_test_image(input_image);

    // Display the input image for debugging
    xil_printf("Generated test image (28x28):\r\n");
    print_feature_map(input_image, INPUT_SIZE, INPUT_SIZE, "Input Image");

    // Initialize fully connected weights with random values
    // In a real application, these would be loaded from trained model
    xil_printf("Initializing FC weights...\r\n");
    for (int i = 0; i < FC_NEURONS; i++)
    {
        for (int j = 0; j < 12 * 12 * CONV1_FILTERS; j++)
        {
            fc_weights[i][j] = (rand() % 21) - 10; // Random values -10 to 10
        }
    }

    // Run the complete CNN inference pipeline
    int predicted_class;
    int result = run_cnn_inference(input_image, &predicted_class);

    // Display results
    if (result == 0)
    {
        xil_printf("\r\nCNN Inference Completed Successfully!\r\n");
        xil_printf("Predicted Class: %d\r\n", predicted_class);
    }
    else
    {
        xil_printf("CNN Inference Failed!\r\n");
    }

    // Cleanup allocated memory and hardware resources
    free(input_image);
    cleanup_npu();

    return result;
}

//=============================================================================
// NPU Hardware Initialization
//=============================================================================

int init_npu(void)
{
#ifdef USE_LINUX_MMAP
    void *map_base;

    // Open /dev/mem for direct hardware access on Linux
    if ((mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1)
    {
        xil_printf("Cannot open /dev/mem\r\n");
        return -1;
    }

    // Map NPU registers into user space
    map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                    mem_fd, NPU_BASE_ADDR & ~MAP_MASK);
    if (map_base == (void *)-1)
    {
        xil_printf("Cannot map NPU registers\r\n");
        close(mem_fd);
        return -1;
    }

    // Calculate actual NPU register pointer
    npu_mem = (volatile uint32_t *)((char *)map_base + (NPU_BASE_ADDR & MAP_MASK));
#else
// Bare-metal: Set up memory attributes for NPU access
#ifndef __PPC__ // Skip for PowerPC architectures
    // Configure memory region as non-cacheable for hardware registers
    Xil_SetTlbAttributes(NPU_BASE_ADDR, NORM_NONCACHE);
#endif
#endif

    xil_printf("NPU initialized at address 0x%08X\r\n", NPU_BASE_ADDR);

    // Test NPU accessibility by reading status register
    uint32_t test_val = read_npu_reg(NPU_STATUS_REG);
    xil_printf("NPU Status Register: 0x%08X\r\n", test_val);

    return 0;
}

//=============================================================================
// NPU Hardware Cleanup
//=============================================================================

void cleanup_npu(void)
{
#ifdef USE_LINUX_MMAP
    // Unmap NPU registers and close file descriptor
    if (npu_mem != NULL)
    {
        munmap((void *)((char *)npu_mem - (NPU_BASE_ADDR & MAP_MASK)), MAP_SIZE);
        npu_mem = NULL;
    }
    if (mem_fd != -1)
    {
        close(mem_fd);
        mem_fd = -1;
    }
#endif
    // Bare-metal: No cleanup needed for direct memory access
}

//=============================================================================
// NPU Register Access Functions
//=============================================================================

// Write a 32-bit value to NPU register
int write_npu_reg(uint32_t offset, uint32_t value)
{
#ifdef USE_LINUX_MMAP
    if (npu_mem == NULL)
        return -1;
    *(npu_mem + (offset >> 2)) = value; // Divide by 4 for word addressing
#else
    // Use Xilinx I/O functions for direct register access
    Xil_Out32(NPU_BASE_ADDR + offset, value);
#endif
    return 0;
}

// Read a 32-bit value from NPU register
uint32_t read_npu_reg(uint32_t offset)
{
#ifdef USE_LINUX_MMAP
    if (npu_mem == NULL)
        return 0;
    return *(npu_mem + (offset >> 2)); // Divide by 4 for word addressing
#else
    // Use Xilinx I/O functions for direct register access
    return Xil_In32(NPU_BASE_ADDR + offset);
#endif
}

//=============================================================================
// Test Image Generation
//=============================================================================

void generate_test_image(data_t *image)
{
    // Generate a simple pattern resembling the digit "7"
    // Initialize all pixels to 0 (black background)
    memset(image, 0, INPUT_SIZE * INPUT_SIZE * sizeof(data_t));

    // Draw horizontal line at top (top stroke of "7")
    for (int j = 5; j < 23; j++)
    {
        image[2 * INPUT_SIZE + j] = 255; // Row 2, columns 5-22
        image[3 * INPUT_SIZE + j] = 255; // Row 3, columns 5-22 (thicker line)
    }

    // Draw diagonal line (diagonal stroke of "7")
    for (int i = 4; i < 25; i++)
    {
        // Calculate diagonal position: starts at column 22, slopes down-left
        int j = 22 - (i - 4) * 15 / 21;
        if (j >= 0 && j < INPUT_SIZE)
        {
            image[i * INPUT_SIZE + j] = 255; // Main diagonal pixel
            if (j > 0)
                image[i * INPUT_SIZE + j - 1] = 255; // Thicker line
        }
    }

    // Add random noise to make image more realistic
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++)
    {
        if (image[i] == 0 && (rand() % 20) == 0)
        {                           // 5% chance for background pixels
            image[i] = rand() % 50; // Low-intensity noise
        }
    }
}

//=============================================================================
// Debug Output Function
//=============================================================================

void print_feature_map(data_t *data, int height, int width, const char *name)
{
    xil_printf("\r\n%s (%dx%d):\r\n", name, height, width);

    // Print first 10 rows and 20 columns to avoid overwhelming output
    for (int i = 0; i < height && i < 10; i++)
    {
        for (int j = 0; j < width && j < 20; j++)
        {
            xil_printf("%4d ", data[i * width + j]); // 4-digit field width
        }
        if (width > 20)
            xil_printf("..."); // Indicate truncation
        xil_printf("\r\n");
    }
    if (height > 10)
        xil_printf("...\r\n"); // Indicate truncation
}

//=============================================================================
// Main CNN Inference Pipeline
//=============================================================================

int run_cnn_inference(data_t *input_image, int *output_class)
{
    xil_printf("\r\nStarting CNN Inference...\r\n");

    //=========================================================================
    // Layer 1: Convolution (28x28 -> 24x24x4)
    //=========================================================================

    xil_printf("Layer 1: Convolution (5x5 kernel, 4 filters)\r\n");
    int conv1_out_size = INPUT_SIZE - CONV1_KERNEL + 1; // 28-5+1 = 24

    // Use static buffer instead of malloc to avoid memory allocation issues
    data_t *conv1_output = conv1_output_static_buffer;

    // Clear the static buffer (good practice)
    memset(conv1_output, 0, CONV1_OUTPUT_ELEMENTS * sizeof(data_t));

    // Process each convolution filter
    for (int f = 0; f < CONV1_FILTERS; f++)
    {
        xil_printf("Processing filter %d...\r\n", f);

        // Configure NPU for convolution operation
        write_npu_reg(NPU_CONFIG_REG,
                      (CONV1_FILTERS << 28) |    // Number of channels
                          (CONV1_KERNEL << 20) | // Kernel size
                          (INPUT_SIZE << 12) |   // Input width
                          (INPUT_SIZE << 4) |    // Input height
                          0);                    // Operation type (0 = convolution)

        // Perform convolution in software (hardware would use DMA)
        for (int i = 0; i < conv1_out_size; i++)
        {
            for (int j = 0; j < conv1_out_size; j++)
            {
                acc_t sum = 0;

                // Convolve 5x5 kernel with input region
                for (int ki = 0; ki < CONV1_KERNEL; ki++)
                {
                    for (int kj = 0; kj < CONV1_KERNEL; kj++)
                    {
                        sum += input_image[(i + ki) * INPUT_SIZE + (j + kj)] *
                               conv1_weights[f][ki][kj];
                    }
                }

                // Add bias and store result
                sum += conv1_bias[f];
                conv1_output[f * conv1_out_size * conv1_out_size + i * conv1_out_size + j] =
                    (data_t)(sum >> 8); // Scale down to prevent overflow
            }
        }

        // Start NPU processing (for hardware acceleration)
        write_npu_reg(NPU_CONTROL_REG, 0x1);

        // Platform-specific delay while NPU processes
#if defined(_WIN32) || defined(_WIN64)
        Sleep(1); // 1 millisecond delay on Windows
#elif defined(__linux__)
        usleep(1000); // 1000 microseconds = 1ms on Linux
#else
        // Xilinx-specific delay (if usleep_A9 is available)
        if (Xil_In32(NPU_BASE_ADDR + NPU_STATUS_REG) == 0)
        {
            // usleep_A9(1000); // Uncomment if function is available
        }
#endif

        // Wait for NPU completion with timeout
        uint32_t status = read_npu_reg(NPU_STATUS_REG);
        int timeout = 1000; // Timeout counter
        while ((status & 0x2) == 0 && timeout > 0)
        { // Wait for completion bit
#if defined(_WIN32) || defined(_WIN64)
            Sleep(0); // Yield CPU on Windows
#elif defined(__linux__)
            usleep(100); // 100 microsecond delay on Linux
#else
            // Xilinx-specific delay
            if (Xil_In32(NPU_BASE_ADDR + NPU_STATUS_REG) == 0)
            {
                // usleep_A9(100); // Uncomment if function is available
            }
#endif
            status = read_npu_reg(NPU_STATUS_REG);
            timeout--;
        }

        // Check for timeout
        if (timeout == 0)
        {
            xil_printf("NPU timeout for filter %d\r\n", f);
        }

        // Clear NPU control register
        write_npu_reg(NPU_CONTROL_REG, 0x0);
    }

    // Apply ReLU activation function (remove negative values)
    software_relu(conv1_output, conv1_out_size * conv1_out_size * CONV1_FILTERS);

    // Debug output - show first filter's feature map
    xil_printf("Conv1 output sample:\r\n");
    print_feature_map(conv1_output, conv1_out_size, conv1_out_size, "Conv1 Feature Map (Filter 0)");

    //=========================================================================
    // Layer 2: Max Pooling (24x24x4 -> 12x12x4)
    //=========================================================================

    xil_printf("\r\nLayer 2: Max Pooling (2x2)\r\n");
    int pool1_out_size = conv1_out_size / POOL1_SIZE; // 24/2 = 12

    // Allocate memory for pooling output (consider making static if memory is tight)
    data_t *pool1_output = (data_t *)malloc(pool1_out_size * pool1_out_size * CONV1_FILTERS * sizeof(data_t));
    if (pool1_output == NULL)
    {
        xil_printf("Failed to allocate pool1_output\r\n");
        return -1;
    }

    // Process each filter's feature map
    for (int f = 0; f < CONV1_FILTERS; f++)
    {
        // Configure NPU for pooling operation
        write_npu_reg(NPU_CONFIG_REG,
                      (CONV1_FILTERS << 28) |      // Number of channels
                          (POOL1_SIZE << 20) |     // Pooling window size
                          (conv1_out_size << 12) | // Input width
                          (conv1_out_size << 4) |  // Input height
                          1);                      // Operation type (1 = pooling)

        // Perform max pooling in software
        for (int i = 0; i < pool1_out_size; i++)
        {
            for (int j = 0; j < pool1_out_size; j++)
            {
                // Start with first pixel in 2x2 window
                data_t max_val = conv1_output[f * conv1_out_size * conv1_out_size +
                                              (i * 2) * conv1_out_size + (j * 2)];

                // Find maximum value in 2x2 window
                for (int pi = 0; pi < POOL1_SIZE; pi++)
                {
                    for (int pj = 0; pj < POOL1_SIZE; pj++)
                    {
                        data_t val = conv1_output[f * conv1_out_size * conv1_out_size +
                                                  (i * 2 + pi) * conv1_out_size + (j * 2 + pj)];
                        if (val > max_val)
                            max_val = val;
                    }
                }

                // Store maximum value
                pool1_output[f * pool1_out_size * pool1_out_size + i * pool1_out_size + j] = max_val;
            }
        }

        // NPU processing steps (similar to convolution)
        write_npu_reg(NPU_CONTROL_REG, 0x1);

        // Wait for completion with timeout
        uint32_t status = read_npu_reg(NPU_STATUS_REG);
        int timeout = 1000;
        while ((status & 0x2) == 0 && timeout > 0)
        {
#if defined(_WIN32) || defined(_WIN64)
            Sleep(0);
#elif defined(__linux__)
            usleep(100);
#else
            if (Xil_In32(NPU_BASE_ADDR + NPU_STATUS_REG) == 0)
            {
                // usleep_A9(100); // Uncomment if function is available
            }
#endif
            status = read_npu_reg(NPU_STATUS_REG);
            timeout--;
        }
        write_npu_reg(NPU_CONTROL_REG, 0x0);
    }

    // Debug output - show first filter's pooled feature map
    xil_printf("Pool1 output sample:\r\n");
    print_feature_map(pool1_output, pool1_out_size, pool1_out_size, "Pool1 Feature Map (Filter 0)");

    //=========================================================================
    // Layer 3: Fully Connected (576 -> 10)
    //=========================================================================

    xil_printf("\r\nLayer 3: Fully Connected (%d neurons)\r\n", FC_NEURONS);

    // Allocate memory for FC output (small allocation - 10 neurons)
    data_t *fc_output = (data_t *)malloc(FC_NEURONS * sizeof(data_t));
    int fc_input_size = pool1_out_size * pool1_out_size * CONV1_FILTERS; // 12*12*4 = 576

    if (fc_output == NULL)
    {
        xil_printf("Failed to allocate fc_output\r\n");
        free(pool1_output);
        return -1;
    }

    // Compute fully connected layer (matrix multiplication)
    for (int n = 0; n < FC_NEURONS; n++)
    {
        acc_t sum = 0;

        // Multiply flattened input by weights
        for (int i = 0; i < fc_input_size; i++)
        {
            sum += pool1_output[i] * fc_weights[n][i];
        }

        // Add bias and store result
        sum += fc_bias[n];
        fc_output[n] = (data_t)(sum >> 8); // Scale down
    }

    // Apply ReLU activation to final output
    software_relu(fc_output, FC_NEURONS);

    // Display final classification scores
    xil_printf("FC output:\r\n");
    for (int i = 0; i < FC_NEURONS; i++)
    {
        xil_printf("Class %d: %d\r\n", i, fc_output[i]);
    }

    // Find the class with highest score
    *output_class = argmax(fc_output, FC_NEURONS);

    //=========================================================================
    // Performance and Architecture Summary
    //=========================================================================

    xil_printf("\r\nCNN Architecture Summary:\r\n");
    xil_printf("Input: %dx%d = %d pixels\r\n", INPUT_SIZE, INPUT_SIZE, INPUT_SIZE * INPUT_SIZE);
    xil_printf("Conv1: %dx%dx%d = %d features\r\n", conv1_out_size, conv1_out_size, CONV1_FILTERS,
               conv1_out_size * conv1_out_size * CONV1_FILTERS);
    xil_printf("Pool1: %dx%dx%d = %d features\r\n", pool1_out_size, pool1_out_size, CONV1_FILTERS,
               pool1_out_size * pool1_out_size * CONV1_FILTERS);
    xil_printf("FC: %d neurons\r\n", FC_NEURONS);
    xil_printf("Total parameters: %d\r\n",
               CONV1_FILTERS * CONV1_KERNEL * CONV1_KERNEL + CONV1_FILTERS +
                   FC_NEURONS * fc_input_size + FC_NEURONS);

    // Cleanup dynamically allocated memory
    free(pool1_output);
    free(fc_output);
    // Note: conv1_output is static, so no free() needed

    return 0;
}

//=============================================================================
// Activation Functions
//=============================================================================

// ReLU (Rectified Linear Unit) activation function
// Sets all negative values to zero, keeps positive values unchanged
void software_relu(data_t *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }
}

//=============================================================================
// Utility Functions
//=============================================================================

// Find the index of the maximum value in an array
// Used for classification - returns the predicted class
int argmax(data_t *data, int size)
{
    int max_idx = 0;
    data_t max_val = data[0];

    for (int i = 1; i < size; i++)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

//=============================================================================
// Hardware Testing Functions
//=============================================================================

void test_npu_registers(void)
{
    xil_printf("\r\n=== NPU Register Test ===\r\n");

    // Test basic register access
    uint32_t test_patterns[] = {0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA};
    int num_patterns = sizeof(test_patterns) / sizeof(test_patterns[0]);

    uint32_t registers[] = {NPU_CONTROL_REG, NPU_INPUT_ADDR_REG,
                            NPU_OUTPUT_ADDR_REG, NPU_WEIGHT_ADDR_REG, NPU_CONFIG_REG};
    const char *reg_names[] = {"CONTROL", "INPUT_ADDR", "OUTPUT_ADDR", "WEIGHT_ADDR", "CONFIG"};
    int num_registers = sizeof(registers) / sizeof(registers[0]);

    for (int r = 0; r < num_registers; r++)
    {
        xil_printf("Testing %s register (0x%02X):\r\n", reg_names[r], registers[r]);

        for (int p = 0; p < num_patterns; p++)
        {
            write_npu_reg(registers[r], test_patterns[p]);
            uint32_t readback = read_npu_reg(registers[r]);

            if (readback == test_patterns[p])
            {
                xil_printf("  Pattern 0x%08X: PASS\r\n", test_patterns[p]);
            }
            else
            {
                xil_printf("  Pattern 0x%08X: FAIL (read 0x%08X)\r\n",
                           test_patterns[p], readback);
            }
        }
    }

    // Test status register (read-only)
    xil_printf("Testing STATUS register (0x%02X) - Read Only:\r\n", NPU_STATUS_REG);
    uint32_t status = read_npu_reg(NPU_STATUS_REG);
    xil_printf("  Current status: 0x%08X\r\n", status);
}

// Timer function for performance measurement
uint32_t get_timer_value(void)
{
#ifdef __linux__
    static uint32_t counter = 0; // Simple mock for Linux if not using a proper timer
    return counter += rand() % 1000;
#elif defined(_WIN32) || defined(_WIN64)
    // For Windows, GetTickCount() or QueryPerformanceCounter() could be used.
    // This is a simple placeholder.
    static DWORD last_tick = 0;
    if (last_tick == 0)
        last_tick = GetTickCount();
    DWORD current_tick = GetTickCount();
    DWORD diff = current_tick - last_tick;
    last_tick = current_tick;
    return diff; // Returns milliseconds elapsed, not a high-resolution timer value.
#else
// Use Xilinx timer for bare-metal
// Ensure XPAR_GLOBAL_TMR_BASEADDR is defined in xparameters.h for this target.
#if defined(XPAR_GLOBAL_TMR_BASEADDR)
    return Xil_In32(XPAR_GLOBAL_TMR_BASEADDR + 0x200); // Global timer counter, offset might vary
#else
    return 0; // Placeholder if timer base address is not defined
#endif
#endif
}
