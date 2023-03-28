package com.example.demo.demos.staticClassTest;

public  class Teacher {
    private String name;
    public static String schoolName="莱阳九中";
    public static final  String NATION="中国";
    public static void teach_study(){
        System.out.println("老师教学生学习");

    }
    public void teach_life(){

        System.out.println(schoolName+"的老师教学生生活");
    }
    public Teacher() {
    }

    public Teacher(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public static String getSchoolName() {
        return schoolName;
    }

    public static void setSchoolName(String schoolName) {
        Teacher.schoolName = schoolName;
    }

    @Override
    public String toString() {
        return "Teacher{" +
                "name='" + name + '\'' +
                '}';
    }
    public void display(){
        System.out.println("schoolName="+schoolName);
    }
}
