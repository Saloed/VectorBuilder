class Main extends Number{
	

public static void main(String[] args) {
		int counter = 0;
		for (String str : args) {
			System.out.println(str);
			counter++;
		}
		System.out.println(counter);
		return ;
	}
}