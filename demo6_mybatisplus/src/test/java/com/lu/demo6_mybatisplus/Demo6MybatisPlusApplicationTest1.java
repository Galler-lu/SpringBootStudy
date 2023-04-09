package com.lu.demo6_mybatisplus;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.StringUtils;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.lu.demo6_mybatisplus.domain.Books;
import com.lu.demo6_mybatisplus.mapper.bookMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@SpringBootTest
public class Demo6MybatisPlusApplicationTest1 {
    @Autowired
    private com.lu.demo6_mybatisplus.mapper.bookMapper bookMapper;
    @Test
    public void test1(){
        //组装查询条件
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        //查询bookID大于110，小于120,且detail1为空，而且bookName包含不要进场字段
        booksQueryWrapper.between("bookID",1010,1020)
                .isNotNull("detail")
                .like("bookName","不要");
        List<Books> books = bookMapper.selectList(booksQueryWrapper);
        for (Books book : books) {
            System.out.println(book);
        }

    }
    @Test
    public void test2(){
        //组装排序条件
        //先按bookCounts升序，若相同再按bookID降序
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        booksQueryWrapper.orderByAsc("bookCounts")
                .orderByDesc("bookID");
        List<Books> books = bookMapper.selectList(booksQueryWrapper);
        for (Books book : books) {
            System.out.println(book);
        }
    }

    @Test
    public void test3(){
        //组装删除条件
        //删除bookName等于真狗
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        booksQueryWrapper.eq("bookName","真狗");
        int delete = bookMapper.delete(booksQueryWrapper);
        if (delete!=0){
            System.out.println("删除成功");
        }else {
            System.out.println("删除失败");
        }
    }
    @Test
    public void test4(){
        //将（bookCounts大于8并且bookName中包含有不要进厂）或detail包含狗都不要进厂
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        booksQueryWrapper.gt("bookCounts",8).like("bookName","不要进厂")
                        .or().like("detail","狗都不进厂");
        Books books = new Books();
        books.setBookName("进厂的缺点");
        books.setDetail("论不进厂的一万个理由");
        int update = bookMapper.update(books, booksQueryWrapper);
        if (update>0){
            System.out.println("更新成功");
        }else {
            System.out.println("更新失败");
        }
    }
    @Test
    public void test5(){
        //查询Books的bookName,bookID,detail
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        booksQueryWrapper.select("bookID","bookID","detail");
        List<Map<String, Object>> maps = bookMapper.selectMaps(booksQueryWrapper);
        maps.forEach(System.out::println);
    }
    @Test
    public void test6(){
        //实现子查询
        //查询bookID小于等于3的书籍信息
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        booksQueryWrapper.inSql("bookID","select bookID from book where bookID<=3");
        List<Books> books = bookMapper.selectList(booksQueryWrapper);
        books.forEach(System.out::println);
    }
    @Test
    public void test7(){
        //类似于xml中的动态sql
        //写法一
        //在真正开发的过程中，组装条件是常见的功能，而这些条件数据来源于用户输入，是可选的，因
        //此我们在组装这些条件时，必须先判断用户是否选择了这些条件，若选择则需要组装该条件，若
        //没有选择则一定不能组装，以免影响SQL执行的结果
        Integer bookID=1010;
        String bookName="";
        Integer bookCounts=5;
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        //StringUtils.isNotBlank()判断某字符串是否不为空且长度不为0且不由空白符(whitespace)
//        构成
        if (StringUtils.isNotBlank(bookName)){
            booksQueryWrapper.like("bookName","进厂");
        }
        if (bookID==1010){
            booksQueryWrapper.gt("bookID",bookID);
        }
        if (bookCounts>4){
            booksQueryWrapper.between("bookCounts",bookCounts,8);
        }
//      Preparing: SELECT bookID,bookName,bookCounts,detail FROM book WHERE (bookID > ? AND bookCounts BETWEEN ? AND ?)
//      Parameters: 1010(Integer), 5(Integer), 8(Integer)
        List<Books> books = bookMapper.selectList(booksQueryWrapper);
        books.forEach(System.out::println);


    }
    @Test
    public void test8(){
        //动态sql写法二
        Integer bookID=1010;
        String bookName="";
        Integer bookCounts=5;
        QueryWrapper<Books> booksQueryWrapper = new QueryWrapper<>();
        booksQueryWrapper.like(StringUtils.isNotBlank(bookName),"bookName","进厂")
                .gt(bookID==1010,"bookID",bookID)
                .between(bookCounts>4,"bookCounts",bookCounts,8);
        List<Books> books = bookMapper.selectList(booksQueryWrapper);
        books.forEach(System.out::println);
    }
    @Test
    public void test9(){
        //动态sql写法三，使用LambdaQueryWrapper
        Integer bookID=1010;
        String bookName="";
        Integer bookCounts=5;
        LambdaQueryWrapper<Books> booksLambdaQueryWrapper = new LambdaQueryWrapper<>();
        booksLambdaQueryWrapper.like(StringUtils.isNotBlank(bookName),Books::getBookName,"进厂")
                .gt(bookID==1010,Books::getBookID,bookID)
                .between(bookCounts>4,Books::getBookCounts,bookCounts,8);
        List<Books> books = bookMapper.selectList(booksLambdaQueryWrapper);
        books.forEach(System.out::println);

    }
    @Test
    public void test10(){
        //分页操作
        //首先去配置分页插件，再使用
        Page<Books> booksPage = new Page<>(1, 3);
        Page<Books> page = bookMapper.selectPage(booksPage, null);
        List<Books> records = page.getRecords();
        System.out.println("--------------------------------------------");
        records.forEach(System.out::println);
        System.out.println("--------------------------------------------");
        System.out.println("总页数："+page.getPages());
        System.out.println("总记录数："+page.getTotal());
        System.out.println("每页显示大小："+page.getSize());
        System.out.println("当前页："+page.getCurrent());
        System.out.println("是否有下一页："+page.hasNext());
        System.out.println("是否前一页:"+page.hasPrevious());
    }
}
