package com.example.demo.demos;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@TableName(value = "book")
public class Book {
//    @TableId(type = IdType.AUTO)
    private Integer id;
    private String name;
    private String type;
    private String description;
}
