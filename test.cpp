int main(int argc, char** argv)
{
    while(argc!=0)
    {
        argc-=1;
        printf(argv[argc]);
    }
    return argc;
}

void bubbleSort(int arr[], int n)
{
      int swapped = true;
      int j = 0;
      int tmp;
      while (swapped) {
            swapped = false;
            j++;
            for (int i = 0; i < n - j; i++) {
                  if (arr[i] > arr[i + 1]) {
                        tmp = arr[i];
                        arr[i] = arr[i + 1];
                        arr[i + 1] = tmp;
                        swapped = true;
                  }
            }
      }
}