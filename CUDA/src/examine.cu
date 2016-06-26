#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LSIZE 31
#define MIN_LIM 12.0
#define MAX_LIM 30.0

void check_input(int argc,char* argv[]);
__global__ void examine(float *d_coordinates,int *d_coords_within,int d_lines);
long calc_lines(char *filename);

int main(int argc,char * argv[])
{
  check_input(argc,argv);             // Check cmd inputs
  char *filename=argv[3];             // Variable initialization
  int coll = atoi(argv[1]);
  int exec_time=atoi(argv[2]);
  int threads=atoi(argv[4]);
  int BLOCKSIZE = atoi(argv[5]);
  long loop_count;
  loop_count =calc_lines(filename);						// Count the lines of input file
  FILE *input=fopen(filename,"r");                        // Open file with file descriptor
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);                       // Get gpu's properties information
  if(coll != -1)													// Handle max_collisions argument
  {
    if(coll>loop_count)
    {
      printf("[!] Warning: Specified collisions to be tested exceed the ones in input file\n");
      printf("[!] Setting the number of collisions to the maximum (taken from input file)\n");
    }
    else
    {
      if (coll<0) return 1;
      loop_count = coll;
    }
  }
  if (BLOCKSIZE==-1)                      // Handle blocksize argument
  {
      BLOCKSIZE=512;                      // A default value
  }
  else
  {
    if (BLOCKSIZE%prop.warpSize!=0 || BLOCKSIZE<=0)
    {
      printf("[-]Block_size must be a positive multiple of gpu's warp_size %d \n",prop.warpSize );
      return 5;
    }
  }
  if (threads!=-1)                        // Handle threads argument
  {
    if (threads<=0) return 4;
    if (threads%BLOCKSIZE!=0)
    {
      threads=(threads/BLOCKSIZE)*BLOCKSIZE;
    }
  }
  else
  {
    threads=prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount;
  }
  // Print some information [ Usefull for debugging ]
  printf("[+] GPU-model: %s\tTotal GPU memory %ld MB \n",prop.name,prop.totalGlobalMem/(1024*1024) );
  printf("[!] You are trying to allocate %ld MBs of memmory on CPU-RAM and GPU-GlobalMem\n",threads*3*sizeof(float)/(1024*1024) );
  printf("[+] Launching %d GPU-Threads with BlockSize %d\n",threads,BLOCKSIZE );
  // Initialize CUDA WallClock-time counters as events
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 blockSize(BLOCKSIZE);              // Declare CUDA Block size explicitly
  dim3 gridSize(threads/BLOCKSIZE);       // Declare CUDA Grid size explicitly
  float *h_coordinates=(float * )malloc(3*threads*sizeof(float));    // allocate Host memmory for elements to be read from file
  float *d_coordinates;
  int *d_coords_within,*h_coords_within=(int*)malloc(sizeof(int));    // allocate Host memmory for the counter of coordinates in area of interest
  *h_coords_within=0;
                                                          // Allocate memmory on CUDA capable Device for:
  cudaMalloc(&d_coordinates,3*threads*sizeof(float));     // input file's coordinates
  cudaMalloc(&d_coords_within,sizeof(int));               // coordinates counter

  cudaMemcpy(d_coords_within,h_coords_within,sizeof(int),cudaMemcpyHostToDevice);   // Initialize the value of cuounter on Device
  int i,j=0;
  float time_elapsed = 0;
  printf("[+] Working...\n" );
  cudaEventRecord(start);           // Starting time reference
  while(j<loop_count && (exec_time==-1?1:time_elapsed<exec_time))               // Main loop of the programm
  {
    if (j+threads>loop_count)
    {
      threads=loop_count-j;
      cudaFree(d_coordinates);
      cudaMalloc(&d_coordinates,3*threads*sizeof(float));
    }
    for(i=0;i<threads;i++)
    {
      fscanf(input,"%f %f %f",&h_coordinates[i*3],&h_coordinates[i*3+1],&h_coordinates[i*3+2]);   // Read cooordinates from file
    }
    cudaMemcpy(d_coordinates,h_coordinates,3*threads*sizeof(float),cudaMemcpyHostToDevice);       // Copy read cooordinates on Device
    examine<<<gridSize,blockSize>>>(d_coordinates,d_coords_within,3*threads);                     // Launch gpu kernel for calculations
    cudaEventRecord(stop);                               // Stop time reference
    cudaEventSynchronize(stop);                         // Block CPU until "stop" event is recorded
    cudaEventElapsedTime(&time_elapsed, start, stop);  // Calculate the time elapsed in milliseconds
    time_elapsed=time_elapsed/1000;                    // Convert milliseconds to seconds
    j+=threads;
  }
  // Destroy CUDA timers
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(h_coords_within,d_coords_within,sizeof(int),cudaMemcpyDeviceToHost);   // Copy results from Device to Host

  //Printing results
  printf("[+] Main part of the program was being executed for :: %.3f :: sec)\n", time_elapsed);
  printf("[+] %ld coordinates have been analyzed\n[+] %d cooordinates were inside the area of interest\n[+] %ld coordinates read per second\n", loop_count, *h_coords_within, (time_elapsed<1?loop_count:loop_count/(int)time_elapsed));

  // Free Host and Device memory
  cudaFree(d_coordinates);
  cudaFree(d_coords_within);
  fclose(input);
  free(h_coordinates);
  free(h_coords_within);

  return 0;
}
__global__ void examine(float *d_coordinates,int *d_coords_within,int d_lines)
{
    int index=blockIdx.x*3*blockDim.x+3*threadIdx.x;                                                 // find the index of starting element for each thread on each block
    float coord1=d_coordinates[index],coord2=d_coordinates[index+1],coord3=d_coordinates[index+2];  // Copy cooordinates from GPU's global memory to thread's local memory
    if(index>=d_lines) return;
    if(coord1 >= MIN_LIM && coord1 <= MAX_LIM && coord2 >= MIN_LIM && coord2 <= MAX_LIM && coord3 >= MIN_LIM && coord3 <= MAX_LIM)
    {
      	                                                       // If the current coordinate is within the accepted limits,
              atomicAdd((unsigned int*)d_coords_within,1);    // So as threads do not mess up the values
    }
}
void check_input(int argc,char *argv[]) 																		// Handle number of arguments errors and show usage
{
	if (argc<6 || argc>6)
	{
		printf("[-] Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] [1D_blockSize]\nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help"))
    {
      printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of gpu-threads to use / # Rows in memmory\n1D_blocksize: gpu-blocksize to use" );
      printf("\t ======Usefull info!======\n");
      printf("1) 1D_blockSize must be a multiple of 32. (or whatever warp_size is supported by your GPU)\n2) Threads should be a multiple of blockSize\n 3)These 2 parameters are important for performance\n" );
    }
		exit(2);
	}
}
long calc_lines(char *filename) 																						// Calculates the lines of input file
{
	FILE *file=fopen(filename,"r");
	fseek(file,0L,SEEK_END);															//set file position indicator right to the end-of-file
	long lines=ftell(file);																//store the number of bytes since the beginning of the file
	fseek(file,0L,SEEK_SET);
	fclose(file);
	return lines/LSIZE;																			//return lines count of the file
}
