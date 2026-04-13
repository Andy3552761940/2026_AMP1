<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>
<%@ page import="java.text.DecimalFormat" %>
<%
request.setCharacterEncoding("UTF-8");
String resultText = null;
String riskLevel = null;
String recommendation = null;

String ageParam = request.getParameter("age");
String hbParam = request.getParameter("hb");
String gender = request.getParameter("gender");
String symptoms = request.getParameter("symptoms");

if ("POST".equalsIgnoreCase(request.getMethod()) && ageParam != null && hbParam != null) {
    try {
        int age = Integer.parseInt(ageParam.trim());
        double hb = Double.parseDouble(hbParam.trim());

        // 简化演示逻辑：仅用于 GUI 样例展示
        double baseRisk = 0.0;
        if (hb < 90) {
            baseRisk += 0.6;
        } else if (hb < 110) {
            baseRisk += 0.35;
        } else if (hb < 130) {
            baseRisk += 0.15;
        }

        if (age >= 65) {
            baseRisk += 0.2;
        } else if (age >= 45) {
            baseRisk += 0.1;
        }

        if ("female".equals(gender)) {
            baseRisk += 0.05;
        }

        if ("yes".equals(symptoms)) {
            baseRisk += 0.1;
        }

        double finalRisk = Math.min(0.99, baseRisk);
        DecimalFormat df = new DecimalFormat("0.00%");
        resultText = df.format(finalRisk);

        if (finalRisk >= 0.6) {
            riskLevel = "高风险";
            recommendation = "建议尽快复查血常规并结合临床进一步评估。";
        } else if (finalRisk >= 0.3) {
            riskLevel = "中风险";
            recommendation = "建议近期进行复检，并关注饮食与休息。";
        } else {
            riskLevel = "低风险";
            recommendation = "当前风险较低，建议定期随访。";
        }
    } catch (Exception e) {
        request.setAttribute("error", "输入格式有误，请检查年龄和血红蛋白值。");
    }
}
%>
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>贫血风险 GUI 演示</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="container">
    <h1>贫血风险评估 GUI（JSP）</h1>
    <p class="subtitle">可直接部署到支持 JSP 的服务器（如 Tomcat）</p>

    <% if (request.getAttribute("error") != null) { %>
    <div class="alert error"><%= request.getAttribute("error") %></div>
    <% } %>

    <form method="post" action="index.jsp" class="card">
        <div class="form-row">
            <label for="age">年龄</label>
            <input id="age" name="age" type="number" min="0" max="120" required value="<%= ageParam == null ? "" : ageParam %>">
        </div>

        <div class="form-row">
            <label for="hb">血红蛋白 Hb (g/L)</label>
            <input id="hb" name="hb" type="number" step="0.1" min="0" max="250" required value="<%= hbParam == null ? "" : hbParam %>">
        </div>

        <div class="form-row">
            <label for="gender">性别</label>
            <select id="gender" name="gender" required>
                <option value="male" <%= "male".equals(gender) ? "selected" : "" %>>男</option>
                <option value="female" <%= "female".equals(gender) ? "selected" : "" %>>女</option>
            </select>
        </div>

        <div class="form-row">
            <label for="symptoms">是否有头晕/乏力</label>
            <select id="symptoms" name="symptoms" required>
                <option value="no" <%= "no".equals(symptoms) ? "selected" : "" %>>否</option>
                <option value="yes" <%= "yes".equals(symptoms) ? "selected" : "" %>>是</option>
            </select>
        </div>

        <button type="submit">开始评估</button>
    </form>

    <% if (resultText != null) { %>
    <div class="card result">
        <h2>评估结果</h2>
        <p><strong>风险概率：</strong><%= resultText %></p>
        <p><strong>风险等级：</strong><%= riskLevel %></p>
        <p><strong>建议：</strong><%= recommendation %></p>
        <p class="note">注：该页面为演示 GUI，结果不作为医疗诊断依据。</p>
    </div>
    <% } %>
</div>
</body>
</html>
