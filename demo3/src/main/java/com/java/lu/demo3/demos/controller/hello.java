package com.java.lu.demo3.demos.controller;
//package com.java.lu.demo3.demos.controller;
//
import com.java.lu.demo3.MyDataSource;
import com.java.lu.demo3.MyDataSource1;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class hello {

   @RequestMapping("/hello1")
    public String hello(){
        System.out.println("hello word");
        return "hello,world";
    }
    @Value("${server.port}")
    private String port;
    @Value("${likes[0]}")
    private String likes;
   @Value("${users[0].name}")
   private String name;
    @GetMapping("/hello2")
    public String hello2(){
        System.out.println("likes为："+likes);
        System.out.println("users.name为："+name);
        System.out.println("server.port为："+port);
       return "hello2";
    }

    //测试使用${属性名}
    @Value("${baseDir}")
    private String baseDir;
    @Value("${center1}")
    private String center1;
    @Value("${center3}")
    private String center3;
    @Value("${center4}")
    public String center4;
    @PostMapping("hello3")
    public String hello3(){
        System.out.println("baseDir为："+baseDir);
        System.out.println("center1为："+center1);
        System.out.println("center3为："+center3);
        System.out.println("center4为："+center4);
        return "hello3";
    }
    //数据读取的优化之将配置文件的数据封装到Environment中进行读取
    @Autowired
    private Environment environment;
    @PostMapping("/hello4")
    public String hello4(){
        System.out.println("port为："+environment.getProperty("server.port"));
        System.out.println("users.name为："+environment.getProperty("users[0].name"));
        System.out.println("likes为："+environment.getProperty("likes[0]"));
        System.out.println("center1为："+environment.getProperty("center1"));
        return "hello4";
    }
    @Autowired
    private MyDataSource myDataSource;
    @Autowired
    private MyDataSource1 myDataSource1;
    @PostMapping("hello5")
    public String hello5(){
        System.out.println(myDataSource.getDriver());
        System.out.println(myDataSource.getPassword());
        System.out.println(myDataSource.getUrl());
        System.out.println(myDataSource.getUsername());
        System.out.println(myDataSource1.toString());
        return "hello5";
    }

}

