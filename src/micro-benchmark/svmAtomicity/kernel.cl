__kernel void CLRunner(__global	int * input){

	 int i,j;
	 for (i = 0; i < 10; i++) {
	     int temp = input;  
	     for (j = 0; j < 1000; j++) {}
//	     delay(10000);
	 }
}
