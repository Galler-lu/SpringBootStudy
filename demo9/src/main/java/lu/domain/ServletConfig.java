package lu.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.convert.DataSizeUnit;
import org.springframework.boot.convert.DurationUnit;
import org.springframework.stereotype.Component;
import org.springframework.util.unit.DataSize;
import org.springframework.util.unit.DataUnit;
import org.springframework.validation.annotation.Validated;

import javax.validation.constraints.Max;
import java.time.Duration;
import java.time.temporal.ChronoUnit;

@Data
@AllArgsConstructor
@NoArgsConstructor
//@Component
@ConfigurationProperties(prefix = "servlets")
@Validated
public class ServletConfig {
//    @Value("${servlets.ipAddress}")
    private String ipAddress;
//    @Value("${servlets.port}")
    @Max(value = 8080,message = "端口号不能超过100")
    private Integer port;
//    @Value("${servlets.timeout}")
    private long timeout;
    @DurationUnit(ChronoUnit.HOURS)
    private Duration time;
    @DataSizeUnit(DataUnit.GIGABYTES)
    private DataSize dataSize;
}
