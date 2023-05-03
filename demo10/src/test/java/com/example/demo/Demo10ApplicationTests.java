package com.example.demo;

import com.example.demo.demos.Book;
import com.example.demo.service.bookService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.RequestBuilder;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.test.web.servlet.ResultMatcher;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders;
import org.springframework.test.web.servlet.result.ContentResultMatchers;
import org.springframework.test.web.servlet.result.MockMvcResultMatchers;
import org.springframework.test.web.servlet.result.StatusResultMatchers;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT )
@AutoConfigureMockMvc
class Demo10ApplicationTests {

    @Test
    void contextLoads(@Autowired MockMvc mockMvc) throws Exception {
        RequestBuilder requestBuilder= MockMvcRequestBuilders.get("/getByID");
        mockMvc.perform(requestBuilder);

    }
    @Test
    public void test1(@Autowired MockMvc mockMvc) throws Exception {
        //测试状态
        RequestBuilder requestBuilder= MockMvcRequestBuilders.get("/getByID1");
        StatusResultMatchers status = MockMvcResultMatchers.status();
        ResultMatcher ok = status.isOk();
        ResultActions perform = mockMvc.perform(requestBuilder);
        perform.andExpect(ok);
    }

    //测试响应体非json格式
    @Test
    public void test2(@Autowired MockMvc mockMvc) throws Exception {
        RequestBuilder requestBuilder=MockMvcRequestBuilders.get("/getByID");
        ResultActions perform = mockMvc.perform(requestBuilder);
        ContentResultMatchers content = MockMvcResultMatchers.content();
        ResultMatcher springboot = content.string("springboot1");
        perform.andExpect(springboot);


    }
    //匹配json格式
    @Test
    public void test3(@Autowired MockMvc mockMvc) throws Exception {

        RequestBuilder request=MockMvcRequestBuilders.get("/books");
        ResultActions perform = mockMvc.perform(request);
        ContentResultMatchers content = MockMvcResultMatchers.content();
        ResultMatcher json = content.json("{\"id\":1002,\"name\":\"测试\",\"type\":\"必读\",\"description\":\"仅做测试\"}");
        perform.andExpect(json);
    }


}
