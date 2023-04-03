package com.example.demo.demos.study02;

import org.junit.Test;

public class arrTest {
//    @Test
//    public void test1(){
//        int[] ints = {1, 2};
//        System.out.println(ints[0]);
//        String name[];
//        name=new String[5];
//        name[0]="鲁文慧";
//        String sex[]={"男","女"};
//    }
    public static void main(String[] args) {
        try {
            String[] strings = {"1","2","3","4"};
            for (int i = 0; i < 5; i++) {
                System.out.println(strings[i]);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
        }

    }
}
