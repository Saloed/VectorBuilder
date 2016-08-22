int main(int argc, char** argv)
{
    if(argc != 0)
    {
        //std::cout<<argv[0]<<std::endl;
        int* arr = new int[argc];
        for(int  i = 0; i<argc; i++)
        {
            arr[i] = i;
        }
        return 0;
    }
    else
    {
        return -1;
    }
}