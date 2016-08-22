class SpecialType
{
public:
    int field_1;
    char** field_2;
    SpecialType(int first, char** second)
    {
        field_1 = first;
        field_2 = second;
    }

    int is_ready(void);
    int special_fun(void);
}

int main(int argc, char** argv)
{
    SpecialType t = new SpecialType(argc,argv)
    if (t.is_ready())
    {
        return t.special_fun()
    }
    else
    {
        return -1;
    }
}