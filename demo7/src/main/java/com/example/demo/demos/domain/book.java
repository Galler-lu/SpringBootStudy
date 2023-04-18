package com.example.demo.demos.domain;

import com.baomidou.mybatisplus.annotation.TableId;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class book {
//    @TableId("id")
    private Integer id;
    private String name;
    private String type;
    private String description;
}
