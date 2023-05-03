package lu;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(properties = {"servlets.port=80"})
class Demo9ApplicationTests {

    @Value("${servlets.port}")
    private String port;
    @Autowired
    private String BeanConfig;

    @Test
    void contextLoads() {
        System.out.println(port);
    }
    @Test
    public void test1(){
        System.out.println(BeanConfig);
    }

}
