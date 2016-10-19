import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import org.reflections.Reflections;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by sobol on 9/28/16.
 */
public class ASTParser {

    public ASTParser() {
    }

    public  int getInt(){
        return 0;
    }

    // private static String getNodeName(Node node) {
    //     String[] nameParts = node.getClass().getName().split("\\.");
    //     return nameParts[nameParts.length - 1];
    // }

}