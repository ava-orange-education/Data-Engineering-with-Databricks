using Demo.Models;
using Microsoft.AspNetCore.Mvc;
using System.Data;
using System.Data.Odbc;

namespace Demo.Controllers
{
    public class DemoController : Controller
    {
        public IActionResult Index()
        {
            ViewBag.Result = string.Empty;
            return View();
        }

        [HttpPost]
        public IActionResult Index(vmQuery query)
        {
            ViewBag.Result = GetData(query.queryText);
            return View();
        }

        public string GetData(string query)
        {
            string result = string.Empty;
            OdbcConnectionStringBuilder odbcConnectionStringBuilder = new OdbcConnectionStringBuilder
            {
                Driver = "Simba Spark ODBC Driver"
                //, //Simba Spark ODBC Driver
                //Dsn = "SimbaSpark_1"
            };
            odbcConnectionStringBuilder.Add("Host", "adb-777777777777777.14.azuredatabricks.net");
            odbcConnectionStringBuilder.Add("Port", "443");
            odbcConnectionStringBuilder.Add("SSL", "1");
            odbcConnectionStringBuilder.Add("ThriftTransport", "2");
            odbcConnectionStringBuilder.Add("AuthMech", "3");
            odbcConnectionStringBuilder.Add("UID", "token");
            odbcConnectionStringBuilder.Add("PWD", "#############################-2");
            odbcConnectionStringBuilder.Add("HTTPPath", "sql/protocolv1/o/########/#######j");


            using (OdbcConnection connection = new OdbcConnection(odbcConnectionStringBuilder.ConnectionString))
            {
                OdbcCommand command = new OdbcCommand(query, connection);
                command.CommandTimeout = 5000000;
                connection.Open();
                OdbcDataReader reader = command.ExecuteReader();
                DataTable data = new DataTable();
                data.Load(reader);
                reader.Close();
                command.Dispose();
                result = ConvertDataTableToHTML(data);
            }
            return result;
        }

        public string ConvertDataTableToHTML(DataTable dt)
        {
            string html = "<table class='table table table-striped table-bordered'>";
            //add header row
            html += "<tr>";
            for (int i = 0; i < dt.Columns.Count; i++)
                html += "<td>" + dt.Columns[i].ColumnName + "</td>";
            html += "</tr>";
            //add rows
            for (int i = 0; i < dt.Rows.Count; i++)
            {
                html += "<tr>";
                for (int j = 0; j < dt.Columns.Count; j++)
                    html += "<td>" + dt.Rows[i][j].ToString() + "</td>";
                html += "</tr>";
            }
            html += "</table>";
            return html;
        }
    }
}
