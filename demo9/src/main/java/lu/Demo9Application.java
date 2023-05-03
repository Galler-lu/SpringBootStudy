package lu;

import lu.domain.ServletConfig;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.ConfigurableApplicationContext;

@SpringBootApplication
@EnableConfigurationProperties(ServletConfig.class)
public class Demo9Application {

    public static void main(String[] args) {
        ConfigurableApplicationContext context = SpringApplication.run(Demo9Application.class, args);
        ServletConfig bean = context.getBean(ServletConfig.class);
        System.out.println(bean);
    }

}
