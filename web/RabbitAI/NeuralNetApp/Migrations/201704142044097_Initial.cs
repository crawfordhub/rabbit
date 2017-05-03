namespace NeuralNetApp.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class Initial : DbMigration
    {
        public override void Up()
        {
            CreateTable(
                "dbo.NeuralNetworks",
                c => new
                    {
                        Id = c.Guid(nullable: false),
                        Name = c.String(),
                        LocationInfo = c.String(),
                    })
                .PrimaryKey(t => t.Id);
            
        }
        
        public override void Down()
        {
            DropTable("dbo.NeuralNetworks");
        }
    }
}
