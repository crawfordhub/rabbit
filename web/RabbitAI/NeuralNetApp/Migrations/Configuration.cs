namespace NeuralNetApp.Migrations
{
	using System;
	using System.Data.Entity;
	using System.Data.Entity.Migrations;
	using System.Linq;
	using NeuralNetApp.Models;

	internal sealed class Configuration : DbMigrationsConfiguration<NeuralNetApp.Models.NeuralNetAppContext>
	{
		public Configuration()
		{
			AutomaticMigrationsEnabled = false;
		}

		protected override void Seed(NeuralNetApp.Models.NeuralNetAppContext context)
		{
			//  This method will be called after migrating to the latest version.
			context.NeuralNetworks.AddOrUpdate(
				new NeuralNetwork
				{
					Id = Guid.NewGuid(),
					Name = "Medical Imaging",
					LocationInfo = "INSERT LOCATION INFO"
				});
		}
	}
}
