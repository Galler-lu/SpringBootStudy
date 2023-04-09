package com.lu.demo6_mybatisplus.domain;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@TableName(value = "book")
//注意此处，若是数据库中将bookID设置为自增则项目构建没有问题。
// 若是没有则需添加@TableId,但不影响不涉及bookID的操作
public class Books {
    //参数value:设置当前实体类字段对应的表中字段；参数type:可以设置逐渐的增加策略(如自增(需要将表中的逐渐也打上自增)或者雪花算法)
    @TableId(type = IdType.AUTO)//将当前注解所对应的字段设置为主键，注意mybatis-plus会默认将id作为主键
    private Integer bookID;
//    @TableField("bookName")//将实体类中的字段对应表中的字段
    private String bookName;
    private Integer bookCounts;
    private String detail;
}
