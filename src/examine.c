#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define MIN_LIM 12.0
#define MAX_LIM 30.0

int check_input(int argc,char *argv[]);
FILE *check_input_file(char *filename);
long calc_time(struct timespec start, struct timespec end, char print_flag);

int main(int argc,char * argv[])
{
	if (check_input(argc,argv)==1)			// Simple argument number checking
	{
		return 1;
	}

	struct timespec start, end;				// Initialize
	long coords_within_lim = 0;				// vars needed for
	long coords_total = 0;					// program configuration						
	int coll = atoi(argv[1]);				// and basic 
	int runtime = atoi(argv[2]);			// calculations			
	int threads_num = atoi(argv[4]);		// in order to get			   
	int proc_num = atoi(argv[5]);			// the desired output
	float coords_val[3] = {0, 0, 0};
	long time_elapsed = 0;

	FILE *input = check_input_file(argv[3]);	// File opening check
	if  (!input)
	{
		return 1;
	}
	clock_gettime(CLOCK_MONOTONIC, &start);		// Initialize time calculation
	while(fscanf(input, "%f %f %f", &coords_val[0], &coords_val[1], &coords_val[2]) != EOF)		// The main loop of the program
	{
		if(coords_val[0] >= MIN_LIM && coords_val[0] <= MAX_LIM && coords_val[1] >= MIN_LIM && coords_val[1] <= MAX_LIM && coords_val[2] >= MIN_LIM && coords_val[2] <= MAX_LIM)
		{
			coords_within_lim++;		// If the current coordinate is within the accepted limits, update the number of accepted coordinates
		}
		coords_total++;					// Update the total number of coordinates read
		if(coords_total == coll)		// If the max number of collisions/coord check specified in the arguments is the same as the current
		{								// coords that have been read, stop the file reading and show the results.
			break;						// (Note that if -1 is specified, this check is redundant)
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end);		// Stop the timer 
	time_elapsed = calc_time(start, end, 1);	// Calculate the time elapsed
	printf("[+] %ld coordinates have been read\n[+] %ld cooordinates were inside the area of interest\n", coords_total, coords_within_lim);

	fclose(input);						// Close the file
	printf("[+] Done! \n" );

	return 0;
}

int check_input(int argc,char *argv[])
{
	if (argc<6 || argc>6) // If input arguments are not as much as meant to be, the algorithm ends indicating it's usage.
	{
		printf("[-] Usage: ./examine [max_collisions] [max_exec_time] [input_file] [Threads] [Processes] \nUse \"-1\": for no boundies \n");
		if (argc==2) if (!strcmp(argv[1],"--help")) printf("max_collisions: Maximum number of collisions\nmax_exec_time: Maximum execution time\ninput_file: Filename to examine\nThreads: Number of threads to use\nProcesses: Number of Processes to use.\n" );
		return 1;
	}
	return 0;
}

FILE *check_input_file(char *filename)
{
	FILE *temp = fopen(filename,"r");
	if (!temp)
	{
		printf("[!] Input file does not exist.\nExiting...\n");
		return NULL;
	}
	return temp;
}

long calc_time(struct timespec start, struct timespec end, char print_flag)	// Function that calculates the time elapsed between start - end
{																			// Returns the time elapsed in seconds for program handling (2nd arg)
	long interval_sec = end.tv_sec - start.tv_sec;							// print_flag is a flag that enables printing
	long interval_nsec = end.tv_nsec - start.tv_nsec;
	if(interval_nsec < 0)
	{
		interval_nsec += 1000000000;
		interval_sec--;
	}
	if(print_flag == 1)
	{
		printf("[+] Main part of the program was being executed for :: %ld.%06ld :: sec)\n", interval_sec, interval_nsec);
	}
	return interval_sec;
}
