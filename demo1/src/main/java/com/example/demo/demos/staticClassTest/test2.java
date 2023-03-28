package com.example.demo.demos.staticClassTest;

import com.example.demo.demos.ExtendClass.Person;
import org.junit.Test;

public class test2 {
    @Test
    public void test(){
        System.out.println(Teacher.schoolName);
        Teacher.setSchoolName("青科");
        Teacher teacher1 = new Teacher("卢晨");
        teacher1.display();
        teacher1.schoolName="哈商";

        Teacher teacher2 = new Teacher("陆沉");
        teacher2.display();
        Teacher teacher3 = new Teacher("陆晨");
        teacher3.display();
    }
}
