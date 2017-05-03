/*--------------------------------------------------------------------------------------
 * Copyright (c) 2017 Rabbit AI. All Rights Reserved.
 *
 *------------------------------------------------------------------------------------*/

/*
 * @author Taylor Dean
 */

using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Data.Entity.Infrastructure;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Http;
using System.Web.Http.Description;
using NeuralNetApp.Models;

namespace NeuralNetApp.Controllers
{
	public class NeuralNetworksController : ApiController
	{
		private NeuralNetAppContext db = new NeuralNetAppContext();

		// GET: api/NeuralNetworks
		public IQueryable<NeuralNetwork> GetNeuralNetworks()
		{
			return db.NeuralNetworks;
		}

		// GET: api/NeuralNetworks/5
		[ResponseType(typeof(NeuralNetwork))]
		public async Task<IHttpActionResult> GetNeuralNetwork(Guid id)
		{
			NeuralNetwork neuralNetwork = await db.NeuralNetworks.FindAsync(id);
			if (neuralNetwork == null)
			{
				return NotFound();
			}

			return Ok(neuralNetwork);
		}

		// PUT: api/NeuralNetworks/5
		[ResponseType(typeof(void))]
		public async Task<IHttpActionResult> PutNeuralNetwork(Guid id, NeuralNetwork neuralNetwork)
		{
			if (!ModelState.IsValid)
			{
				return BadRequest(ModelState);
			}

			if (id != neuralNetwork.Id)
			{
				return BadRequest();
			}

			db.Entry(neuralNetwork).State = EntityState.Modified;

			try
			{
				await db.SaveChangesAsync();
			}
			catch (DbUpdateConcurrencyException)
			{
				if (!NeuralNetworkExists(id))
				{
					return NotFound();
				}
				else
				{
					throw;
				}
			}

			return StatusCode(HttpStatusCode.NoContent);
		}

		// POST: api/NeuralNetworks
		[Authorize]
		[ResponseType(typeof(NeuralNetwork))]
		public async Task<IHttpActionResult> PostNeuralNetwork(NeuralNetwork neuralNetwork)
		{
			if (!ModelState.IsValid)
			{
				return BadRequest(ModelState);
			}

			if (neuralNetwork.Id == null)
			{
				neuralNetwork.Id = Guid.NewGuid();
			}

			db.NeuralNetworks.Add(neuralNetwork);

			try
			{
				await db.SaveChangesAsync();
			}
			catch (DbUpdateException)
			{
				if (NeuralNetworkExists(neuralNetwork.Id))
				{
					return Conflict();
				}
				else
				{
					throw;
				}
			}

			return CreatedAtRoute("DefaultApi", new { id = neuralNetwork.Id }, neuralNetwork);
		}

		// DELETE: api/NeuralNetworks/5
		[ResponseType(typeof(NeuralNetwork))]
		public async Task<IHttpActionResult> DeleteNeuralNetwork(Guid id)
		{
			NeuralNetwork neuralNetwork = await db.NeuralNetworks.FindAsync(id);
			if (neuralNetwork == null)
			{
				return NotFound();
			}

			db.NeuralNetworks.Remove(neuralNetwork);
			await db.SaveChangesAsync();

			return Ok(neuralNetwork);
		}

		protected override void Dispose(bool disposing)
		{
			if (disposing)
			{
				db.Dispose();
			}
			base.Dispose(disposing);
		}

		private bool NeuralNetworkExists(Guid id)
		{
			return db.NeuralNetworks.Count(e => e.Id == id) > 0;
		}
	}
}