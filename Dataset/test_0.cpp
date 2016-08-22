int t_0 (int arg1, char arg2, int arg3)
{
    int t = fun2( fun1 (arg1,arg2) , arg3);
    if (t >= 0)
    {
        return fun2( fun1 (arg1, arg2) , arg3);
    }
    else
    {
        return 0;
    }

}