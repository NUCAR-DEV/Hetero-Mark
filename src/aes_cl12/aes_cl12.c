//
//  main.c
//  CasAES_CL
//
//  Created by Carter McCardwell on 12/7/14.
//  Copyright (c) 2014 Casdidicus. All rights reserved.
//

#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)
#define Nb 4
#define Nr 14
#define Nk 8

#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

#include <CL/cl.h>

uint8_t s[256] = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};

uint8_t Rcon[256] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
    0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
    0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
    0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
    0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
    0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
    0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
    0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
    0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
    0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
    0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
    0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
    0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
    0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
    0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
    0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d
};

int RUNNING_THREADS = 1;
//RUNNING_THREADS is autoset to number of compute units, 1 is failsafe

uint32_t expanded_key[60] = { 0x00 };

char* stradd(const char* a, const char* b){
    size_t len = strlen(a) + strlen(b);
    char *ret = (char*)malloc(len * sizeof(char) + 1);
    *ret = '\0';
    return strcat(strcat(ret, a) ,b);
}

uint32_t rw(uint32_t word)
{
    union {
        uint8_t bytes[4];
        uint32_t word;
    } subWord  __attribute__ ((aligned));
    subWord.word = word;
    
    uint8_t B0 = subWord.bytes[3], B1 = subWord.bytes[2], B2 = subWord.bytes[1], B3 = subWord.bytes[0];
    subWord.bytes[3] = B1; //0
    subWord.bytes[2] = B2; //1
    subWord.bytes[1] = B3; //2
    subWord.bytes[0] = B0; //3
    
    return subWord.word;
}

uint32_t sw(uint32_t word)
{
    union {
        uint32_t word;
        uint8_t bytes[4];
    } subWord  __attribute__ ((aligned));
    subWord.word = word;
    
    subWord.bytes[3] = s[subWord.bytes[3]];
    subWord.bytes[2] = s[subWord.bytes[2]];
    subWord.bytes[1] = s[subWord.bytes[1]];
    subWord.bytes[0] = s[subWord.bytes[0]];
    
    return subWord.word;
}

void K_Exp(uint8_t* pk)
{
    int i = 0;
    union {
        uint8_t bytes[4];
        uint32_t word;
    } temp __attribute__ ((aligned));
    union {
        uint8_t bytes[4];
        uint32_t word;
    } univar[100] __attribute__ ((aligned));
    
    for (i = 0; i < Nk; i++)
    {
        univar[i].bytes[3] = pk[i*4];
        univar[i].bytes[2] = pk[i*4+1];
        univar[i].bytes[1] = pk[i*4+2];
        univar[i].bytes[0] = pk[i*4+3];
    }
    
    for (i = Nk; i < Nb*(Nr+1); i++)
    {
        temp.word = univar[i-1].word;
        if (i % Nk == 0)
        {
            temp.word = (sw(rw(temp.word)));
            temp.bytes[3] = temp.bytes[3] ^ (Rcon[i/Nk]);
        }
        else if (Nk > 6 && i % Nk == 4)
        {
            temp.word = sw(temp.word);
        }
        if (i-4 % Nk == 0)
        {
            temp.word = sw(temp.word);
        }
        univar[i].word = univar[i-Nk].word ^ temp.word;
    }
    for (i = 0; i < 60; i++)
    {
        expanded_key[i] = univar[i].word; //printf("\n%i : %x", i, univar[i].word);
    }
}

int main(int argc, const char * argv[])
{
  //     printf("CasAES_CL Hyperthreaded AES-256 Encryption for OpenCL 1.2 processors - compiled 12/8/2014 Rev. 1\nCarter McCardwell, Northeastern University NUCAR - http://coe.neu.edu/~cmccardw - mccardwell.net\nPlease Wait...");
    
    clock_t c_start, c_stop;
    c_start = clock();
    
    FILE *infile;
    FILE *keyfile;
    FILE *outfile;
    FILE *cl_code;
    
    infile = fopen(argv[2], "r");
    if (infile == NULL) { printf("error_in"); return(1); }
    keyfile = fopen(argv[3], "rb");
    if (keyfile == NULL) { printf("error_key"); return(1); }
    outfile = fopen(argv[4], "w");
    if (outfile == NULL) { printf("error (permission error: run with sudo or in directory the user owns)"); return (1); }
    
    //Hex info, or ASCII
    int hexMode = 1;
    if (strcmp(argv[1], "h") == 0) { hexMode = 1; }
    else if (strcmp(argv[1], "a") == 0) { hexMode = 0; }
    else { printf("error: first argument must be \'a\' for ASCII interpretation or \'h\' for hex interpretation\n"); return(1); }
    
    uint8_t key[32];
    
    for (int i = 0; i < 32; i++)
    {
        fscanf(keyfile, "%x", &key[i]);
    }
    
    //Calculate expanded key and add to kernel source
    K_Exp(&key);
    
    char *append_str = "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n#define Nb 4\n#define Nr 14\n#define Nk 8\n\n__constant uint eK[60]={";
    for (int i = 0; i < 60; i++)
    {
        char *key_element = (char *)malloc(sizeof(uint32_t));
        //sprintf(key_element, "%x", expanded_key[i]);
        append_str = stradd(append_str, "0x");
        append_str = stradd(append_str, key_element);
        if (i != 59) { append_str = stradd(append_str, ","); }
    }
    append_str = stradd(append_str, "};\n");
    
    cl_code = fopen("aes_cl12_Kernels.cl", "r");
    char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
    fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
    fclose(cl_code);
    
    append_str = stradd(append_str, source_str);
    size_t length = strlen(append_str);
    
    //Set OpenCL Context
    cl_int err;
    cl_platform_id platform;
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) { printf("platformid"); }
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) { printf("getdeivceid %i", err); }
    
    cl_uint numberOfCores;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores), &numberOfCores, NULL);
    if (numberOfCores > 1) { RUNNING_THREADS = numberOfCores; }
    printf("\nRunning with %i compute units", RUNNING_THREADS);
    
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { printf("context"); }
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) { printf("queue"); }
    
    cl_mem key_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t)*32, &key, &err);
    if (err != CL_SUCCESS) { printf("memcpy_key"); }
    
    cl_program program = clCreateProgramWithSource(context, 1, &append_str, &length, &err);
    if (err != CL_SUCCESS) { printf("createprogram"); }
    
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        
        // Get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        
        // Print the log
        //printf("%s\n", log);
    }
    
    //Generic platform-agnostic dispatch <iThreadDSP 1> <CPU/HPTCP Symol>
    uint8_t states[16 * RUNNING_THREADS];
    for (int i = 0; i < 16*RUNNING_THREADS; i++) { states[i] = 0x00; }
    int ch = 0;
    int spawn = 0;
    int end = 1;
    int currentOffset = -1;
    while (end)
    {
        spawn = 0;
        for (int i = 0; i < RUNNING_THREADS; i++) //Dispatch many control threads that will report back to main (for now 5x) - 1 worker per state
        {
            currentOffset = i*16;
            spawn++;
            for (int ix = 0; ix < 16; ix++)
            {
                if (hexMode == 1)
                {
                    if (fscanf(infile, "%x", &states[currentOffset+ix]) != EOF) { ; }
                    else
                    {
                        if (ix > 0) { for (int ixx = ix; ixx < 16; ixx++) { states[currentOffset+ixx] = 0x00; } }
                        else { spawn--; }
                        i = RUNNING_THREADS + 1;
                        end = 0;
                        break;
                    }
                }
                else
                {
                    ch = getc(infile);
                    if (ch != EOF) { states[currentOffset+ix] = ch; }
                    else
                    {
                        if (ix > 0) { for (int ixx = ix; ixx < 16; ixx++) { states[currentOffset+ixx] = 0x00; } }
                        else { spawn--; }
                        i = RUNNING_THREADS + 1;
                        end = 0;
                        break;
                    }
                }
            }
        }
        //arrange data correctly
        for (int i = 0; i < spawn; i++)
        {
            currentOffset = i*16;
            uint8_t temp[16];
            memcpy(&temp[0], &states[currentOffset], sizeof(uint8_t));
            memcpy(&temp[4], &states[currentOffset+1], sizeof(uint8_t));
            memcpy(&temp[8], &states[currentOffset+2], sizeof(uint8_t));
            memcpy(&temp[12], &states[currentOffset+3], sizeof(uint8_t));
            memcpy(&temp[1], &states[currentOffset+4], sizeof(uint8_t));
            memcpy(&temp[5], &states[currentOffset+5], sizeof(uint8_t));
            memcpy(&temp[9], &states[currentOffset+6], sizeof(uint8_t));
            memcpy(&temp[13], &states[currentOffset+7], sizeof(uint8_t));
            memcpy(&temp[2], &states[currentOffset+8], sizeof(uint8_t));
            memcpy(&temp[6], &states[currentOffset+9], sizeof(uint8_t));
            memcpy(&temp[10], &states[currentOffset+10], sizeof(uint8_t));
            memcpy(&temp[14], &states[currentOffset+11], sizeof(uint8_t));
            memcpy(&temp[3], &states[currentOffset+12], sizeof(uint8_t));
            memcpy(&temp[7], &states[currentOffset+13], sizeof(uint8_t));
            memcpy(&temp[11], &states[currentOffset+14], sizeof(uint8_t));
            memcpy(&temp[15], &states[currentOffset+15], sizeof(uint8_t));
            for (int c = 0; c < 16; c++) { memcpy(&states[currentOffset+c], &temp[c], sizeof(uint8_t)); }
        }
        //Set data for workers----------
        
        cl_mem dev_states;
        cl_int status = CL_SUCCESS;
        
        dev_states = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 16*spawn*sizeof(uint8_t), states, &status);
        if (status != CL_SUCCESS || dev_states == NULL) { printf("\nclCreateBuffer: %i", status); }
        
        //status = clEnqueueWriteBuffer(queue, dev_states, CL_TRUE, 0, 16*spawn*sizeof(uint8_t), &states, 0, NULL, NULL);
        if (status != CL_SUCCESS) { printf("\nclEnqueueWriteBuffer: %i", status); }
        
        cl_kernel aesKernel = clCreateKernel(program, "CLRunnerntrl", &status);
        if (status != CL_SUCCESS) { printf("\nclCreateKernel: %i", status); }
        //status = clSetKernelArg(aesKernel, 0, 16*spawn*sizeof(uint8_t), &dev_states);
        status = clSetKernelArg(aesKernel, 0, sizeof(cl_mem), &dev_states);
        if (status != CL_SUCCESS) { printf("\nclSetKernelArg: %i", status); }
        
        const size_t local_ws = 1;
        const size_t global_ws = spawn;
        cl_event event;
        status = clEnqueueNDRangeKernel(queue, aesKernel, 1, NULL, &global_ws, NULL, 0, NULL, &event);
        if (status != CL_SUCCESS) { printf("\nclEnqueueNDRangeKernel: %i", status); }
        
        clWaitForEvents(1, &event);
        
        status = clEnqueueReadBuffer(queue, dev_states, CL_TRUE, 0, 16*spawn*sizeof(uint8_t), &states, 0, NULL, NULL);
        if (status != CL_SUCCESS) { printf("\nclEnqueueReadBuffer: %i", status); }
        
        clReleaseMemObject(dev_states);
        
        for (int i = 0; i < spawn; i++)
        {
            currentOffset = i*16;
            for (int ix = 0; ix < 4; ix++)
            {
                char hex[3];
                //sprintf(hex, "%x", states[currentOffset+ix]);
                for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
                //sprintf(hex, "%x", states[currentOffset+ix+4]);
                for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
                //sprintf(hex, "%x", states[currentOffset+ix+8]);
                for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
                //sprintf(hex, "%x", states[currentOffset+ix+12]);
                for (int i = 0; i < 3; i++) { putc(hex[i], outfile); }
            }
        }
    }
    c_stop = clock();
    float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC ) * 1000;
    
    printf("Done - Time taken: %f ms\n", diff);
    fclose(infile);
    fclose(outfile);
    fclose(keyfile);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    return 0;
}
