# Vercel Deployment Guide

This branch (`vercel-deploy`) is optimized for Vercel deployment.

## 🚀 Deploy to Vercel

### Option 1: One-Click Deploy
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Jenverse/Langcache-full-app-deploy/tree/vercel-deploy)

### Option 2: Manual Deploy

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Clone this branch**
   ```bash
   git clone -b vercel-deploy https://github.com/Jenverse/Langcache-full-app-deploy.git
   cd Langcache-full-app-deploy
   ```

3. **Deploy**
   ```bash
   vercel --prod
   ```

## ⚙️ Configuration

### Required Settings
After deployment, users need to configure:

1. **Visit your deployed app**
2. **Go to Settings tab**
3. **Enter credentials**:
   - **OpenAI API Key**: `sk-proj-...`
   - **Redis URL**: `redis://default:password@host:port`

### Redis Setup
For production, use Redis Cloud:
- [Redis Cloud](https://redis.com/redis-enterprise-cloud/)
- [Upstash Redis](https://upstash.com/) (Vercel integration)

## 🔧 Vercel-Specific Changes

This branch includes:
- ✅ Serverless function optimization
- ✅ Static file routing
- ✅ Production environment detection
- ✅ Increased Lambda size limit
- ✅ Extended function timeout

## 🔒 Security

- **No secrets in code**: All credentials provided by users
- **Environment variables**: Not required (user-provided settings)
- **HTTPS**: Automatic with Vercel
- **CORS**: Configured for production

## 📊 Performance

- **Cold start**: ~2-3 seconds
- **Warm requests**: ~200-500ms
- **Static assets**: CDN cached
- **Function timeout**: 30 seconds

## 🐛 Troubleshooting

### Common Issues

1. **Function timeout**: Increase in vercel.json if needed
2. **Redis connection**: Ensure Redis URL is accessible from Vercel
3. **API limits**: Check OpenAI API quotas
4. **CORS errors**: Check domain configuration

### Logs
```bash
vercel logs <deployment-url>
```

## 🔄 Updates

To update the deployment:
1. Make changes to this branch
2. Push to GitHub
3. Vercel auto-deploys

## 📞 Support

For deployment issues:
- Check Vercel dashboard
- Review function logs
- Test locally first
